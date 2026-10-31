import copy
import math
import random


# ---------------------------------------------------------------- 独立 oracle
# 结构独立：oracle 先把所有分片拼成一个 (N x D) 的全局矩阵，用**离线 softmax**
# （先求全行最大值，再一次性 exp 求和）算输出；traces 则是对每一步**从头重算**
# 已经可见的全部 key（不做 rescale，也就没有 corr）；sources 由降序环绕切片拼出；
# skipped 由「块内最小 key 位置 > 最大 query 位置」判定；comm 由尾部补零的列表拼出。
# 参考解是逐 rank、逐 step 的在线递推（corr 乘累积量），两条路径不共用中间量。
def _dot(a, b):
    total = 0.0
    for index in range(len(a)):
        total += a[index] * b[index]
    return total


def _oracle(q_shards, k_shards, v_shards, causal, scale, dtype_bytes):
    ranks = len(q_shards)
    length = len(q_shards[0])
    width = len(q_shards[0][0])
    total = ranks * length
    Q = [list(row) for shard in q_shards for row in shard]
    K = [list(row) for shard in k_shards for row in shard]
    V = [list(row) for shard in v_shards for row in shard]

    rows_out = []
    for p in range(total):
        scores = []
        for j in range(total):
            if causal and j > p:
                continue
            scores.append((scale * _dot(Q[p], K[j]), j))
        top = max(score for score, _ in scores)
        denom = 0.0
        acc = [0.0] * width
        for score, j in scores:
            weight = math.exp(score - top)
            denom += weight
            for d in range(width):
                acc[d] += weight * V[j][d]
        rows_out.append([value / denom for value in acc])
    out = [[rows_out[rank * length + i] for i in range(length)] for rank in range(ranks)]

    # traces：每一步都把「到目前为止可见的全部 key」从头重算（不做 rescale、没有 corr）
    traces = []
    for rank in range(ranks):
        rank_traces = []
        previous = [(-math.inf, 0.0)] * length
        seen = []
        for step in range(ranks):
            seen.append((rank - step) % ranks)
            per_row = []
            for i in range(length):
                p = rank * length + i
                keys = []
                for block in seen:
                    stop = block * length + length - 1
                    if causal and stop > p:
                        stop = p
                    for j in range(block * length, stop + 1):
                        keys.append(j)
                if not keys:
                    per_row.append(previous[i])
                    continue
                top = max(scale * _dot(Q[p], K[j]) for j in keys)
                denom = 0.0
                for j in keys:
                    denom += math.exp(scale * _dot(Q[p], K[j]) - top)
                per_row.append((top, denom))
            rank_traces.append(per_row)
            previous = per_row
        traces.append(rank_traces)

    # 第 r 个 rank 的 source 序列：r, r-1, ..., 0, C-1, ..., r+1
    sources = [list(range(rank, -1, -1)) + list(range(ranks - 1, rank, -1)) for rank in range(ranks)]
    skipped = []
    for rank in range(ranks):
        low_of = {source: source * length for source in range(ranks)}
        last_query = rank * length + length - 1
        skipped.append([causal and low_of[source] > last_query for source in sources[rank]])

    payload = 2 * length * width * dtype_bytes
    comm = [[payload] * (ranks - 1) + [0] for _ in range(ranks)]
    return out, traces, sources, skipped, comm


# ---------------------------------------------------------------- 比对工具
def _close(got, want, tol=1e-12):
    if got == want:
        return True
    if math.isinf(got) or math.isinf(want):
        return False
    return abs(got - want) <= tol * max(1.0, abs(got), abs(want))


def _verify(solve, args):
    q, k, v, causal, scale, dtype_bytes = args
    q_before, k_before, v_before = copy.deepcopy(q), copy.deepcopy(k), copy.deepcopy(v)
    got = solve(copy.deepcopy(q), copy.deepcopy(k), copy.deepcopy(v), causal, scale, dtype_bytes)
    assert isinstance(got, tuple) and len(got) == 5, "solve must return a 5-tuple"
    out, traces, sources, skipped, comm = got
    ranks = len(q_before)
    length = len(q_before[0])
    width = len(q_before[0][0])
    want_out, want_traces, want_sources, want_skipped, want_comm = _oracle(
        q_before, k_before, v_before, causal, scale, dtype_bytes
    )

    assert len(out) == ranks and len(traces) == ranks, "one entry per rank"
    assert len(sources) == ranks and len(skipped) == ranks and len(comm) == ranks

    for rank in range(ranks):
        assert len(out[rank]) == length, "out shard row count"
        for i in range(length):
            assert len(out[rank][i]) == width, "out row width"
            for d in range(width):
                value = out[rank][i][d]
                assert isinstance(value, float), "out entries must be floats"
                assert math.isfinite(value), "out must stay finite"
                assert _close(value, want_out[rank][i][d]), f"out[{rank}][{i}][{d}]"
        assert len(sources[rank]) == ranks and len(skipped[rank]) == ranks and len(comm[rank]) == ranks
        previous_max = [-math.inf] * length
        for step in range(ranks):
            assert type(sources[rank][step]) is int, "sources must hold ints"
            assert sources[rank][step] == want_sources[rank][step], "source order"
            assert type(skipped[rank][step]) is bool, "skipped must hold bools"
            assert skipped[rank][step] == want_skipped[rank][step], "skip decision"
            assert type(comm[rank][step]) is int, "comm must hold ints"
            assert comm[rank][step] == want_comm[rank][step], "comm account"
            assert len(traces[rank][step]) == length, "trace row count"
            for i in range(length):
                pair = traces[rank][step][i]
                assert len(pair) == 2, "each trace is an (m, l) pair"
                live_max, live_sum = pair
                assert _close(live_max, want_traces[rank][step][i][0]), f"trace m[{rank}][{step}][{i}]"
                assert _close(live_sum, want_traces[rank][step][i][1]), f"trace l[{rank}][{step}][{i}]"
                assert live_max >= previous_max[i] - 1e-9, "the running max may only grow"
                assert live_sum >= 1.0 - 1e-12, "the running denominator is at least the max term"
                if skipped[rank][step]:
                    assert step > 0, "the rank's own shard is never skipped"
                    assert pair == traces[rank][step - 1][i], "a skipped block leaves the state alone"
                previous_max[i] = live_max
    assert (q, k, v) == (q_before, k_before, v_before), "inputs must not be mutated"
    return out, traces, sources, skipped, comm


# ---------------------------------------------------------------- 样例
def check_example(solve):
    z = [[[0.0], [0.0]], [[0.0], [0.0]]]
    k = [[[1.0], [2.0]], [[3.0], [4.0]]]
    out, traces, sources, skipped, comm = _verify(solve, (z, k, k, True, 1.0, 4))
    assert out == [[[1.0], [1.5]], [[2.0], [2.5]]], out
    assert traces == [
        [[(0.0, 1.0), (0.0, 2.0)], [(0.0, 1.0), (0.0, 2.0)]],
        [[(0.0, 1.0), (0.0, 2.0)], [(0.0, 3.0), (0.0, 4.0)]],
    ], traces
    assert sources == [[0, 1], [1, 0]]
    assert skipped == [[False, True], [False, False]]
    assert comm == [[16, 0], [16, 0]]

    q2 = [[[1.0]], [[1.0]]]
    k2 = [[[1.0]], [[2.0]]]
    out, traces, sources, skipped, comm = _verify(solve, (q2, k2, k2, True, 1.0, 4))
    assert _close(out[0][0][0], 1.0) and _close(out[1][0][0], 1.7310585786300049), out
    assert _close(traces[1][0][0][0], 2.0) and traces[1][0][0][1] == 1.0, traces
    assert _close(traces[1][1][0][1], 1.3678794411714423), traces
    assert sources == [[0, 1], [1, 0]] and skipped == [[False, True], [False, False]]
    assert comm == [[8, 0], [8, 0]]

    out, traces, sources, skipped, comm = _verify(solve, (z, k, k, False, 1.0, 4))
    assert out == [[[2.5], [2.5]], [[2.5], [2.5]]], out
    assert skipped == [[False, False], [False, False]]
    assert traces[0] == [[(0.0, 2.0), (0.0, 2.0)], [(0.0, 4.0), (0.0, 4.0)]], traces[0]
    assert comm == [[16, 0], [16, 0]]

    q4 = [[[0.0]], [[0.0]], [[0.0]]]
    k4 = [[[1.0]], [[2.0]], [[4.0]]]
    out, traces, sources, skipped, comm = _verify(solve, (q4, k4, k4, True, 1.0, 2))
    assert _close(out[0][0][0], 1.0) and _close(out[1][0][0], 1.5)
    assert _close(out[2][0][0], 7.0 / 3.0), out
    assert sources == [[0, 2, 1], [1, 0, 2], [2, 1, 0]]
    assert skipped == [[False, True, True], [False, False, True], [False, False, False]]
    assert comm == [[4, 4, 0], [4, 4, 0], [4, 4, 0]]
    assert [traces[2][step][0][1] for step in range(3)] == [1.0, 2.0, 3.0]


check_example.example = True


# ---------------------------------------------------------------- 对拍
def _random_case(ranks, length, width, seed, low=-2.0, high=2.0):
    rng = random.Random(seed)

    def shards():
        return [
            [[round(rng.uniform(low, high), 3) for _ in range(width)] for _ in range(length)]
            for _ in range(ranks)
        ]

    return shards(), shards(), shards()


def _fixtures():
    cases = []
    for ranks, length, width, causal, seed in [
        (1, 1, 1, True, 11),
        (1, 5, 1, True, 12),
        (1, 4, 3, False, 13),
        (1, 6, 2, False, 14),
        (2, 2, 1, True, 15),
        (2, 3, 2, True, 16),
        (2, 3, 2, False, 17),
        (1, 3, 2, True, 18),
        (3, 1, 1, True, 19),
        (3, 2, 2, True, 20),
        (3, 4, 1, False, 21),
        (3, 3, 3, True, 22),
        (4, 3, 2, True, 23),
        (5, 1, 3, True, 24),
        (5, 2, 1, False, 25),
        (6, 2, 2, True, 26),
        (7, 1, 1, False, 27),
        (8, 1, 1, True, 28),
    ]:
        q, k, v = _random_case(ranks, length, width, seed)
        cases.append((q, k, v, causal, 1.0 / math.sqrt(width), 4))
    return cases


def check_matches_oracle(solve):
    for ranks, length, width, causal, seed in [
        (2, 2, 2, True, 31),
        (3, 2, 1, False, 32),
        (4, 2, 2, True, 33),
    ]:
        q, k, v = _random_case(ranks, length, width, seed)
        for dtype_bytes in (2, 4, 8):
            _verify(solve, (q, k, v, causal, 0.5, dtype_bytes))
    for case in _fixtures():
        _verify(solve, case)


def check_matches_dense_attention(solve):
    import torch

    for ranks, length, width, causal, seed in [
        (2, 3, 2, True, 41),
        (3, 2, 1, True, 42),
        (4, 1, 3, True, 43),
        (2, 4, 2, False, 44),
    ]:
        q, k, v = _random_case(ranks, length, width, seed)
        total = ranks * length
        query = torch.tensor([x for shard in q for row in shard for x in row], dtype=torch.float64)
        key = torch.tensor([x for shard in k for row in shard for x in row], dtype=torch.float64)
        value = torch.tensor([x for shard in v for row in shard for x in row], dtype=torch.float64)
        query = query.reshape(total, width)
        key = key.reshape(total, width)
        value = value.reshape(total, width)
        scores = 0.75 * (query @ key.T)
        if causal:
            scores = scores.masked_fill(
                torch.triu(torch.ones(total, total, dtype=torch.bool), diagonal=1), float("-inf")
            )
        expect = torch.softmax(scores, dim=-1) @ value
        out = _verify(solve, (q, k, v, causal, 0.75, 4))[0]
        flat = [x for shard in out for row in shard for x in row]
        for p in range(total):
            for d in range(width):
                assert _close(flat[p * width + d], expect[p][d].item()), f"row {p} dim {d}"


def check_sources_rotate_backwards(solve):
    for case in _fixtures():
        ranks = len(case[0])
        sources = _verify(solve, case)[2]
        for rank in range(ranks):
            assert sources[rank][0] == rank, "step 0 always reads the rank's own shard"
            assert sorted(sources[rank]) == list(range(ranks)), "each shard arrives exactly once"
            for step in range(1, ranks):
                assert sources[rank][step] == (sources[rank][step - 1] - 1) % ranks, "the ring turns backwards"
    _, _, sources, _, _ = _verify(solve, ([[[1.0]]] * 1, [[[1.0]]] * 1, [[[1.0]]] * 1, True, 1.0, 4))
    assert sources == [[0]]


def check_causal_skip_is_whole_block(solve):
    for ranks, length in [(3, 1), (4, 2), (5, 1)]:
        for causal in (True, False):
            q, k, v = _random_case(ranks, length, 1, 50 + ranks + length)
            _, _, sources, skipped, _ = _verify(solve, (q, k, v, causal, 1.0, 4))
            for rank in range(ranks):
                for step in range(ranks):
                    source = sources[rank][step]
                    expect = causal and source * length > rank * length + length - 1
                    assert skipped[rank][step] is expect, "a block dies only when it starts past the last query"
    # 允许非因果下出现负值、大值：跳过与否只看位置
    q = [[[-1.0]], [[-1.0]], [[-1.0]]]
    k = [[[5.0]], [[5.0]], [[5.0]]]
    assert _verify(solve, (q, k, k, False, 1.0, 4))[3] == [[False] * 3] * 3
    assert _verify(solve, (q, k, k, True, 1.0, 4))[3] == [[False, True, True], [False, False, True], [False] * 3]


def check_uniform_logits_count_visible_keys(solve):
    for ranks, length, causal in [(3, 2, True), (4, 1, True), (1, 5, True), (1, 4, False), (5, 2, False)]:
        q = [[[0.0] for _ in range(length)] for _ in range(ranks)]
        _, k, v = _random_case(ranks, length, 1, 60 + ranks + length)
        out, traces, _, _, _ = _verify(solve, (q, k, v, causal, 3.0, 4))
        flat_v = [row[0] for shard in v for row in shard]
        for rank in range(ranks):
            for i in range(length):
                p = rank * length + i
                visible = p + 1 if causal else ranks * length
                assert _close(traces[rank][ranks - 1][i][1], float(visible)), "logits are all zero"
                expect = sum(flat_v[:visible]) / visible
                assert _close(out[rank][i][0], expect), f"running mean at row {p}"


def check_first_visible_key_survives(solve):
    # 因果注意力下全局第 0 行只有一个可见 key，softmax 权重恰好是 1：输出必须逐位等于 v[0]
    for case in _fixtures():
        if not case[3]:
            continue
        out = _verify(solve, case)[0]
        assert out[0][0] == list(case[2][0][0]), "the very first row sees exactly one key"


def check_skipped_step_keeps_state(solve):
    q, k, v = _random_case(4, 2, 2, 71)
    out, traces, sources, skipped, comm = _verify(solve, (q, k, v, True, 1.0, 4))
    dead = 0
    for rank in range(4):
        for step in range(4):
            if skipped[rank][step]:
                dead += 1
                assert traces[rank][step] == traces[rank][step - 1]
                if step < 3:
                    assert comm[rank][step] == 32, "a skipped block is still received"
                else:
                    assert comm[rank][step] == 0, "nothing is forwarded after the last step"
    assert dead == 6, "C=4 gives 0+1+2+3 dead blocks"
    # 被跳过的块照样计入通信（最后一步没有下一跳要发，所以只见到 C-1 次转发）
    assert comm[0] == [32, 32, 32, 0] and skipped[0] == [False, True, True, True]
    # 非因果时一块都不死，且因果版与它在前 rank 行上必然不同（这里的数值不会碰巧相等）
    assert all(not flag for row in _verify(solve, (q, k, v, False, 1.0, 4))[3] for flag in row)
    assert _verify(solve, (q, k, v, False, 1.0, 4))[0] != out


def check_comm_account(solve):
    for ranks, length, width, dtype_bytes in [(1, 3, 2, 4), (2, 2, 1, 2), (4, 3, 2, 8), (5, 1, 4, 1)]:
        q, k, v = _random_case(ranks, length, width, 80 + ranks)
        payload = 2 * length * width * dtype_bytes
        for causal in (True, False):
            comm = _verify(solve, (q, k, v, causal, 1.0, dtype_bytes))[4]
            assert comm == [[payload] * (ranks - 1) + [0] for _ in range(ranks)], comm
            assert sum(sum(row) for row in comm) == ranks * (ranks - 1) * payload


def check_large_logits(solve):
    # 400 = 20 * 20：不减 max 直接 exp 就会溢出；-400 则要求 max 允许是负数
    q = [[[20.0], [-20.0]]]
    k = [[[20.0], [20.0]]]
    v = [[[1.0], [9.0]]]
    out, traces, _, skipped, _ = _verify(solve, (q, k, v, False, 1.0, 4))
    assert traces[0][0][0] == (400.0, 2.0), traces
    assert traces[0][0][1] == (-400.0, 2.0), traces
    assert _close(out[0][0][0], 5.0) and _close(out[0][1][0], 5.0), out

    # 大数值的随机用例：全程有限
    for ranks, length, width, seed in [(3, 2, 2, 91), (4, 1, 3, 92)]:
        q, k, v = _random_case(ranks, length, width, seed, -50.0, 50.0)
        out = _verify(solve, (q, k, v, True, 30.0, 4))[0]
        assert all(math.isfinite(x) for shard in out for row in shard for x in row)


def check_single_rank(solve):
    q, k, v = _random_case(1, 4, 2, 95)
    out, traces, sources, skipped, comm = _verify(solve, (q, k, v, True, 0.5, 4))
    assert sources == [[0]] and skipped == [[False]] and comm == [[0]]
    assert len(out) == 1 and len(out[0]) == 4
    # 单 rank 的因果注意力就是普通的下三角 softmax
    for i in range(4):
        scores = [0.5 * _dot(q[0][i], k[0][j]) for j in range(i + 1)]
        top = max(scores)
        denom = sum(math.exp(score - top) for score in scores)
        for d in range(2):
            expect = sum(math.exp(scores[j] - top) * v[0][j][d] for j in range(i + 1)) / denom
            assert _close(out[0][i][d], expect)
    assert traces[0][0][0] == (0.5 * _dot(q[0][0], k[0][0]), 1.0), traces


def check_scale(solve):
    for ranks, length, width, causal, seed, scale in [
        (8, 8, 4, True, 101, 0.5),
        (8, 4, 2, False, 102, 1.0),
        (1, 16, 8, True, 103, 0.125),
        (4, 16, 8, True, 104, 2.0),
    ]:
        q, k, v = _random_case(ranks, length, width, seed)
        out, traces, sources, skipped, comm = _verify(solve, (q, k, v, causal, scale, 4))
        assert len(sources[0]) == ranks and len(traces[0][0]) == length
    # 全零输入：输出全零，l 等于可见 key 数（C=8, S=2 共 16 行）
    q = [[[0.0] for _ in range(2)] for _ in range(8)]
    k = [[[0.0] for _ in range(2)] for _ in range(8)]
    v = [[[0.0] for _ in range(2)] for _ in range(8)]
    out, traces, sources, skipped, _ = _verify(solve, (q, k, v, True, 1.0, 4))
    assert all(x == 0.0 for shard in out for row in shard for x in row)
    assert traces[7][7][1][1] == 16.0 and traces[0][7][0][1] == 1.0


def check_invalid(solve):
    def expect_raises(args):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    good = [[[1.0], [2.0]], [[3.0], [4.0]]]
    one = [[[1.0]]]

    # 分片结构
    expect_raises(([], good, good, True, 1.0, 4))
    expect_raises((good, [], good, True, 1.0, 4))
    expect_raises((good, good, [], True, 1.0, 4))
    expect_raises((tuple(good), good, good, True, 1.0, 4))
    expect_raises((good, good[:-1], good, True, 1.0, 4))
    expect_raises((good, good, good + [one], True, 1.0, 4))
    expect_raises((good[:1], good, good, True, 1.0, 4))
    expect_raises(([[]], one, one, True, 1.0, 4))
    expect_raises(([[[]]], one, one, True, 1.0, 4))
    expect_raises(([[[1.0], [1.0, 2.0]]], one, one, True, 1.0, 4))
    expect_raises(([[[1.0]], [[1.0], [2.0]]], one, one, True, 1.0, 4))
    expect_raises(([[[1.0]], [[1.0], [2.0]]], [[[1.0]], [[1.0]]], [[[1.0]], [[1.0]]], True, 1.0, 4))
    expect_raises(([[[1.0]], [[1.0, 2.0]]], [[[1.0]], [[1.0]]], [[[1.0]], [[1.0]]], True, 1.0, 4))
    expect_raises((one, [[[1.0, 2.0]]], [[[1.0]]], True, 1.0, 4))
    expect_raises((one, [[[1.0], [2.0]]], one, True, 1.0, 4))
    expect_raises((one, one, [[[1.0], [2.0]]], True, 1.0, 4))

    # 元素取值
    expect_raises(([[[True]]], one, one, True, 1.0, 4))
    expect_raises(([[["1.0"]]], one, one, True, 1.0, 4))
    expect_raises(([[[None]]], one, one, True, 1.0, 4))
    expect_raises(([[[1 + 2j]]], one, one, True, 1.0, 4))
    expect_raises(([[[float("inf")]]], one, one, True, 1.0, 4))
    expect_raises(([[[float("nan")]]], one, one, True, 1.0, 4))
    expect_raises((one, [[[10 ** 400]]], one, True, 1.0, 4))
    expect_raises((one, one, [[[-float("inf")]]], True, 1.0, 4))

    # 标量参数
    expect_raises((one, one, one, 1, 1.0, 4))
    expect_raises((one, one, one, "yes", 1.0, 4))
    expect_raises((one, one, one, None, 1.0, 4))
    expect_raises((one, one, one, True, 0.0, 4))
    expect_raises((one, one, one, True, -1.0, 4))
    expect_raises((one, one, one, True, True, 4))
    expect_raises((one, one, one, True, "1.0", 4))
    expect_raises((one, one, one, True, float("inf"), 4))
    expect_raises((one, one, one, True, float("nan"), 4))
    expect_raises((one, one, one, True, 10 ** 400, 4))
    expect_raises((one, one, one, True, 1.0, 0))
    expect_raises((one, one, one, True, 1.0, -2))
    expect_raises((one, one, one, True, 1.0, 4.0))
    expect_raises((one, one, one, True, 1.0, True))
    expect_raises((one, one, one, True, 1.0, "4"))

    # 合法边界
    _verify(solve, (one, one, one, True, 1e-300, 1))
    _verify(solve, (one, one, one, False, 1e6, 8))
    _verify(solve, ([[[0]]], [[[-3]]], [[[0]]], True, 1, 16))
    _verify(solve, (good, good, good, True, 1, 2))
