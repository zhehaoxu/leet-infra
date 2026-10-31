import math

import torch


def _oracle(probs, k, capacity, jitter):
    """纯 Python：排序选候选、按 (-key, slot) 定处理顺序、列表记账做贪心占位、fsum 归一化"""
    tokens, experts = probs.shape
    p = probs.tolist()
    noise = jitter.tolist()
    cand = []
    keys = []
    for t in range(tokens):
        row = sorted(range(experts), key=lambda e: (-p[t][e], e))[:k]
        cand.append(row)
        keys.append([p[t][e] + noise[t][e] for e in row])

    slots = tokens * k
    order = sorted(range(slots), key=lambda s: (-keys[s // k][s % k], s))

    remaining = [capacity] * experts
    used = [[] for _ in range(tokens)]
    assign = [[-1] * k for _ in range(tokens)]
    for slot in order:
        t, rank = divmod(slot, k)
        for e in cand[t][rank:]:
            if remaining[e] <= 0 or e in used[t]:
                continue
            assign[t][rank] = e
            remaining[e] -= 1
            used[t].append(e)
            break

    weights = []
    for t in range(tokens):
        raw = [p[t][assign[t][r]] if assign[t][r] >= 0 else 0.0 for r in range(k)]
        denom = math.fsum(raw)
        weights.append([x / denom if denom > 0.0 else 0.0 for x in raw])

    counts = [0] * experts
    for t in range(tokens):
        for e in assign[t]:
            if e >= 0:
                counts[e] += 1
    dropped = sum(1 for t in range(tokens) for e in assign[t] if e < 0)

    make = lambda data, dtype: torch.tensor(data, dtype=dtype)  # noqa: E731
    return (make(assign, torch.int64), make(weights, torch.float64),
            make(counts, torch.int64), dropped, make(order, torch.int64))


def _row_normalize(matrix):
    """就地口径：每行除以自身之和，让随机的非负值看起来像门控概率"""
    return matrix / matrix.sum(dim=1, keepdim=True)


def _case(t=4, e=4, k=2, capacity=2, seed=3, jitter_amp=0.25):
    rng = torch.Generator().manual_seed(seed)
    probs = _row_normalize(torch.rand((t, e), generator=rng, dtype=torch.float64) + 0.05)
    jitter = (torch.rand((t, e), generator=rng, dtype=torch.float64) * 2 - 1) * jitter_amp
    return probs, k, capacity, jitter


def _tie_case(t=8, e=8, k=8, capacity=3, seed=53, levels=4):
    """整数值概率 + 整数值抖动：大量精确并列，把 top-k 与全局键序的并列口径钉死"""
    rng = torch.Generator().manual_seed(seed)
    probs = torch.randint(0, levels, (t, e), generator=rng, dtype=torch.float64)
    jitter = torch.randint(-1, 2, (t, e), generator=rng, dtype=torch.float64)
    return probs, k, capacity, jitter


def _verify(solve, args):
    tensors = [v for v in args if isinstance(v, torch.Tensor)]
    before = [v.clone() for v in tensors]
    expected = _oracle(*args)
    result = solve(*args)
    assert type(result) is tuple and len(result) == 5
    assign, weights, counts, dropped, order = result
    assert type(dropped) is int, type(dropped)
    assert dropped == expected[3], (dropped, expected[3])
    for actual, target in zip((assign, weights, counts, order),
                              (expected[0], expected[1], expected[2], expected[4])):
        assert isinstance(actual, torch.Tensor) and actual.device.type == "cpu"
        assert actual.dtype == target.dtype and actual.shape == target.shape, (
            actual.dtype, actual.shape, target.shape)
        if target.dtype == torch.int64:
            assert torch.equal(actual, target), actual.tolist()
        else:
            torch.testing.assert_close(actual, target, rtol=2e-11, atol=2e-11)
    for value, snapshot in zip(tensors, before):
        assert torch.equal(value, snapshot)
    return result


def _plain_topk(probs, k):
    """不做容量分派的朴素 top-k + renorm，用来核对「没有竞争时结果必须退化为基础口径」"""
    values, indices = _oracle_topk_lists(probs, k)
    weights = []
    for row in values:
        total = math.fsum(row)
        weights.append([v / total if total > 0.0 else 0.0 for v in row])
    make = lambda data, dtype: torch.tensor(data, dtype=dtype)  # noqa: E731
    return make(indices, torch.int64), make(weights, torch.float64)


def _oracle_topk_lists(probs, k):
    experts = probs.shape[1]
    p = probs.tolist()
    values, indices = [], []
    for row in p:
        picked = sorted(range(experts), key=lambda e: (-row[e], e))[:k]
        indices.append(picked)
        values.append([row[e] for e in picked])
    return values, indices


def check_example(solve):
    probs = torch.tensor([
        [0.5, 0.25, 0.125, 0.125],
        [0.4375, 0.3125, 0.125, 0.125],
        [0.375, 0.25, 0.125, 0.25],
        [0.0625, 0.125, 0.25, 0.5]], dtype=torch.float64)
    jitter = torch.zeros(4, 4, dtype=torch.float64)
    jitter[1, 0] = 0.25
    assign, weights, counts, dropped, order = _verify(solve, (probs, 2, 1, jitter))
    # 候选（top-2，并列取小索引）：t0=[0,1] t1=[0,1] t2=[0,1]（0.25 并列时 e1 赢 e3）t3=[3,2]
    # 键：0.6875(1,0) 0.5(0,0) 0.5(3,0) 0.375(2,0) 0.3125(1,1) 0.25(0,1) 0.25(2,1) 0.25(3,1)
    assert order.tolist() == [2, 0, 6, 4, 3, 1, 5, 7]
    assert assign.tolist() == [[1, -1], [0, -1], [-1, -1], [3, 2]]
    assert counts.tolist() == [1, 1, 1, 1]
    assert dropped == 4
    # token0 的 rank0 溢出到 e1（拿到 e1 的概率 0.25），token3 两个槽位都在，0.5/0.75 与 0.25/0.75
    torch.testing.assert_close(weights, torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0]], dtype=torch.float64),
        rtol=2e-11, atol=2e-11)


check_example.example = True


def check_matches_oracle(solve):
    for seed in range(1, 9):
        args = _case(t=2 + seed % 4, e=3 + seed % 3, k=1 + seed % 3,
                     capacity=1 + seed % 3, seed=seed,
                     jitter_amp=[0.0, 0.1, 0.5, 1.5][seed % 4])
        _verify(solve, args)
    # 非连续布局：转置往返、切片视图都不能改变结果
    probs, k, capacity, jitter = _case(t=5, e=4, k=2, capacity=2, seed=31, jitter_amp=0.4)
    _verify(solve, (probs.t().contiguous().t(), k, capacity, jitter))
    _verify(solve, (probs, k, capacity, torch.cat([jitter, jitter], dim=1)[:, :4]))
    _verify(solve, (probs[[0, 1, 2, 3, 4]], k, capacity, jitter))


def check_dispatch_invariants(solve):
    for seed, capacity in ((5, 1), (9, 2), (13, 3), (21, 1)):
        probs, k, _, jitter = _case(t=6, e=5, k=3, capacity=capacity, seed=seed, jitter_amp=0.6)
        args = (probs, k, capacity, jitter)
        assign, weights, counts, dropped, order = _verify(solve, args)
        tokens, experts = probs.shape
        # order 是槽位编号的排列，且沿它方向键值不升
        assert sorted(order.tolist()) == list(range(tokens * k))
        values, indices = _oracle_topk_lists(probs, k)
        keys = [[values[t][j] + float(jitter[t][indices[t][j]]) for j in range(k)]
                for t in range(tokens)]
        flat_keys = [keys[s // k][s % k] for s in order.tolist()]
        assert all(a >= b for a, b in zip(flat_keys, flat_keys[1:])), flat_keys
        # 每专家不超容量；counts 与 assign 一致；丢弃数互补
        assert bool((counts <= capacity).all())
        assert int(counts.sum()) == tokens * k - dropped
        assert [int(v) for v in counts] == [sum(1 for t in range(tokens) for j in range(k)
                                               if int(assign[t][j]) == e) for e in range(experts)]
        for t in range(tokens):
            kept = [int(e) for e in assign[t].tolist() if e >= 0]
            # 同一 token 不能两次落在同一个专家上
            assert len(kept) == len(set(kept)), kept
            for j, e in enumerate(assign[t].tolist()):
                if e < 0:
                    assert float(weights[t][j]) == 0.0
                    continue
                # 分派到的专家必须来自该 token 的候选集，且秩不早于自己
                assert e in indices[t] and indices[t].index(e) >= j
            total = math.fsum(float(v) for v in weights[t].tolist())
            assert total == 0.0 if not kept else abs(total - 1.0) < 1e-12, (t, total)


def check_no_contention_and_full_drop(solve):
    probs, k, _, jitter = _case(t=5, e=4, k=2, capacity=1, seed=11, jitter_amp=0.3)
    # 容量大到不可能有竞争：结果必须退化成朴素 top-k + renorm，且一个都不丢
    assign, weights, counts, dropped, order = _verify(solve, (probs, k, 100, jitter))
    plain_ids, plain_w = _plain_topk(probs, k)
    assert torch.equal(assign, plain_ids)
    torch.testing.assert_close(weights, plain_w, rtol=2e-11, atol=2e-11)
    assert dropped == 0
    assert bool((counts == torch.tensor([sum(1 for t in range(5) for j in range(k)
                                          if int(assign[t][j]) == e) for e in range(4)],
                                        dtype=torch.int64)).all())
    # 容量 0：全丢弃、全零权重，但 order 仍然是合法的处理顺序
    assign, weights, counts, dropped, order = _verify(solve, (probs, k, 0, jitter))
    assert bool((assign == -1).all())
    assert bool((weights == 0.0).all())
    assert bool((counts == 0).all())
    assert dropped == 5 * k


def check_jitter_reorders_and_spills(solve):
    probs = torch.tensor([[0.9, 0.1], [0.8, 0.2]], dtype=torch.float64)
    zero_jitter = torch.zeros(2, 2, dtype=torch.float64)
    # 无抖动：键序 0.9, 0.8, 0.2, 0.1；token1 的 rank0 溢出到 e1，两个 rank1 都被丢
    assign, weights, counts, dropped, order = _verify(solve, (probs, 2, 1, zero_jitter))
    assert order.tolist() == [0, 2, 3, 1]
    assert assign.tolist() == [[0, -1], [1, -1]]
    assert counts.tolist() == [1, 1] and dropped == 2
    torch.testing.assert_close(weights, torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64),
                               rtol=0, atol=0)
    # 给 token0 的 rank1 加正向抖动：它的键反超 rank0，于是先占走 e1，
    # token0 的 rank0 溢出目标被自己占掉，只能丢弃——同一个 token 的幸存者换了位置
    tilted = torch.tensor([[-0.5, 0.5], [0.0, 0.0]], dtype=torch.float64)
    assign, weights, counts, dropped, order = _verify(solve, (probs, 2, 1, tilted))
    assert order.tolist() == [2, 1, 0, 3]
    assert assign.tolist() == [[-1, 1], [0, -1]]
    assert counts.tolist() == [1, 1] and dropped == 2
    torch.testing.assert_close(weights, torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64),
                               rtol=0, atol=0)
    # 负抖动足以把键压到 0：并列时按槽位编号升序处理
    flattening = torch.tensor([[-0.9, -0.1], [-0.8, -0.2]], dtype=torch.float64)
    _, _, _, _, order = _verify(solve, (probs, 2, 1, flattening))
    assert order.tolist() == [0, 1, 2, 3]


def check_degenerate_and_ties(solve):
    # 概率全相等：候选就是 0..k-1，键也相等 -> 处理顺序就是槽位编号
    flat = torch.full((3, 4), 0.25, dtype=torch.float64)
    zero_jitter = torch.zeros(3, 4, dtype=torch.float64)
    assign, weights, counts, dropped, order = _verify(solve, (flat, 2, 1, zero_jitter))
    assert order.tolist() == [0, 1, 2, 3, 4, 5]
    # 容量只有 1：token0 把 e0、e1 各占一个，token1/2 连自己的候选都抢不到
    assert assign.tolist() == [[0, 1], [-1, -1], [-1, -1]]
    assert counts.tolist() == [1, 1, 0, 0] and dropped == 4
    torch.testing.assert_close(weights, torch.tensor(
        [[0.5, 0.5], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float64), rtol=0, atol=0)
    # 整行概率为 0：槽位照样能占位（分派只看候选与容量），但权重必须是 0 而不是 NaN
    probs = torch.zeros(2, 3, dtype=torch.float64)
    assign, weights, counts, dropped, order = _verify(solve, (probs, 2, 3, zero_jitter[:2, :3]))
    assert assign.tolist() == [[0, 1], [0, 1]]
    assert counts.tolist() == [2, 2, 0] and dropped == 0
    assert torch.equal(weights, torch.zeros(2, 2, dtype=torch.float64))
    # 单 token 单专家单槽位
    _verify(solve, (torch.ones(1, 1, dtype=torch.float64), 1, 1, torch.zeros(1, 1, dtype=torch.float64)))
    # k = E：候选是整行，溢出链最长
    probs, k, capacity, jitter = _case(t=3, e=4, k=4, capacity=2, seed=37, jitter_amp=0.9)
    _verify(solve, (probs, 4, capacity, jitter))
    # 64 个槽位、键只有几种整数值：并列必须严格按「秩 / 槽位编号」的口径破，不稳定的排序在这里会暴露
    _verify(solve, _tie_case(t=8, e=8, k=8, capacity=3, seed=53, levels=4))
    _verify(solve, _tie_case(t=8, e=8, k=8, capacity=1, seed=59, levels=2))


def check_invalid(solve):
    probs, k, capacity, jitter = _case(t=3, e=4, k=2, capacity=1, seed=7, jitter_amp=0.2)

    def expect_raises(args, note):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError(f"{note} should raise ValueError")

    expect_raises(([0.5, 0.5], k, capacity, jitter), "probs is not a tensor")
    expect_raises((probs.to(torch.float32), k, capacity, jitter), "probs float32")
    expect_raises((probs, k, capacity, jitter.to(torch.float32)), "jitter float32")
    expect_raises((probs.reshape(-1), k, capacity, jitter.reshape(-1)), "probs 1-D")
    expect_raises((probs, k, capacity, jitter[:, :2]), "jitter shape mismatch")
    expect_raises((torch.zeros(0, 4, dtype=torch.float64), k, capacity,
                   torch.zeros(0, 4, dtype=torch.float64)), "empty batch")
    expect_raises((torch.zeros(3, 0, dtype=torch.float64), 0, capacity,
                   torch.zeros(3, 0, dtype=torch.float64)), "empty experts")
    expect_raises((probs, 0, capacity, jitter), "k = 0")
    expect_raises((probs, 5, capacity, jitter), "k > E")
    expect_raises((probs, True, capacity, jitter), "k is bool")
    expect_raises((probs, 2.0, capacity, jitter), "k is float")
    expect_raises((probs, k, -1, jitter), "negative capacity")
    expect_raises((probs, k, True, jitter), "capacity is bool")
    expect_raises((probs, k, 1.0, jitter), "capacity is float")
    expect_raises((probs * float("nan"), k, capacity, jitter), "nan probs")
    expect_raises((probs - 1.0, k, capacity, jitter), "negative probs")
    expect_raises((probs, k, capacity, jitter + float("inf")), "inf jitter")
    expect_raises((probs, k, capacity, jitter + float("nan")), "nan jitter")
    # 合法边界：capacity 0、k = E、jitter 全 0
    solve(probs, k, 0, jitter)
    solve(probs, 4, 1, torch.zeros_like(jitter))
    solve(probs, 1, 1, torch.zeros_like(jitter))


def check_maximum_shape(solve):
    _verify(solve, _case(t=8, e=8, k=8, capacity=3, seed=41, jitter_amp=1.0))
    _verify(solve, _case(t=8, e=1, k=1, capacity=8, seed=43, jitter_amp=0.0))
    _verify(solve, _case(t=8, e=8, k=8, capacity=1, seed=47, jitter_amp=2.5))
