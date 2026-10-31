import numpy as np

KEYS = {"indices", "distances", "pooled"}

K0 = [[1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [1, -1]]
K1 = [[0, 1], [1, 1], [2, 0], [0, 2], [1, 0], [1, 1]]
Q4 = [
    [[1, 0], [0, 1], [1, 1]],
    [[0, 1], [1, 0], [2, 1]],
    [[1, 1], [0, 1], [0, 1]],
    [[1, 0], [1, 1], [1, 0]],
]


def _oracle(key_states, query_states, num_key_value_groups, topk=1024,
            kernel_size=5, norm=1.0):
    """纯 Python float64 独立实现：interleave 展开 -> 最后一个 query -> 头间求和
    -> 零填充均值池化 -> 值降序、同值下标升序的 top-k。"""
    kv = [[[float(x) for x in v] for v in head] for head in key_states]
    q = [[[float(x) for x in v] for v in head] for head in query_states]
    num_heads, kvh = len(q), len(kv)
    n, d = len(kv[0]), len(kv[0][0])
    keys = []
    for h in range(kvh):
        for _ in range(num_key_value_groups):
            keys.append(kv[h])
    q_last = [head[-1] for head in q]
    sums = []
    for j in range(n):
        s = 0.0
        for h in range(num_heads):
            s += sum(q_last[h][t] * keys[h][j][t] for t in range(d))
        sums.append(s)
    half = kernel_size // 2
    pooled = [
        sum(sums[t] for t in range(max(0, j - half), min(n, j + half + 1))) / kernel_size
        for j in range(n)
    ]
    k_eff = min(topk, n)
    order = sorted(range(n), key=lambda j: (-pooled[j], j))[:k_eff]
    dist = [pooled[j] / norm if norm != 1.0 else pooled[j] for j in order]
    return {"indices": order, "distances": dist, "pooled": pooled}


def _verify(solve, *args, **kwargs):
    k_snap = [[[list(v) for v in head] for head in args[0]]]
    got = solve(*args, **kwargs)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    want = _oracle(*args, **kwargs)
    assert got["indices"] == want["indices"], (got["indices"], want["indices"])
    for a, b in zip(got["pooled"], want["pooled"]):
        assert abs(float(a) - float(b)) < 1e-5, (a, b)
    for a, b in zip(got["distances"], want["distances"]):
        assert abs(float(a) - float(b)) < 1e-5, (a, b)
    assert len(got["distances"]) == len(got["indices"])
    assert args[0] == k_snap[0], "不应修改输入 key_states"
    return got


def check_example(solve):
    """题面示例：GQA 2 组、6 token、kernel 3，手算 sums=[4,4,7,8,5,3]。"""
    got = _verify(solve, [K0, K1], Q4, 2, topk=2, kernel_size=3)
    assert np.allclose(got["pooled"], [8 / 3, 5.0, 19 / 3, 20 / 3, 16 / 3, 8 / 3]), got["pooled"]
    assert got["indices"] == [3, 2], got["indices"]
    assert np.allclose(got["distances"], [20 / 3, 19 / 3])

    got = _verify(solve, [K0, K1], Q4, 2, topk=10, kernel_size=3)
    # topk 超过 n：k_eff=6 全取；j0 与 j5 同值 8/3，按下标升序 j0 在前
    assert got["indices"] == [3, 2, 4, 1, 0, 5], got["indices"]

    got = _verify(solve, [K0, K1], Q4, 2, topk=2, kernel_size=3, norm=2.0)
    assert got["indices"] == [3, 2]
    assert np.allclose(got["distances"], [10 / 3, 19 / 6])


check_example.example = True


def check_matches_oracle(solve):
    """多组合对拍：MHA（groups=1）、kernel 5、topk 截断、负值分数、q_len=1。"""
    import random
    rng = random.Random(42)
    cases = []
    # MHA：8 头各一 KV
    keys8 = [[[rng.randrange(-3, 4) for _ in range(3)] for _ in range(12)] for _ in range(8)]
    qs8 = [[[rng.randrange(-3, 4) for _ in range(3)] for _ in range(2)] for _ in range(8)]
    cases.append((keys8, qs8, 1, dict(topk=5, kernel_size=5)))
    # GQA：4 KV 头 × 3 组
    keys4 = [[[rng.randrange(-2, 3) for _ in range(2)] for _ in range(10)] for _ in range(4)]
    qs4 = [[[rng.randrange(-2, 3) for _ in range(2)] for _ in range(3)] for _ in range(12)]
    cases.append((keys4, qs4, 3, dict(topk=4, kernel_size=5)))
    # 单头单 query、kernel=1（池化恒等）
    keys1 = [[[rng.randrange(-4, 5)] for _ in range(7)]]
    qs1 = [[[rng.randrange(-4, 5)] for _ in range(1)]]
    cases.append((keys1, qs1, 1, dict(topk=3, kernel_size=1)))
    # 负值与零混排
    keysn = [[[-1, 2], [3, -3], [0, 0], [-2, -1], [2, 2], [1, -2], [-1, -1], [3, 1]]]
    qsn = [[[2, -1], [1, 1]]]
    cases.append((keysn, qsn, 1, dict(topk=100, kernel_size=5, norm=0.5)))
    for keys, qs, groups, kw in cases:
        _verify(solve, keys, qs, groups, **kw)


def check_gqa_interleave(solve):
    """repeat_kv 是 repeat_interleave：同组头共用同一份 K；换成 repeat 会得到别的 sums。"""
    got = _verify(solve, [K0, K1], Q4, 2, topk=6, kernel_size=3)
    # interleave 下 sums=[4,4,7,8,5,3]；错误地用 repeat 会得到 [3,5,9,8,7,3]
    assert np.allclose(got["pooled"], [8 / 3, 5.0, 19 / 3, 20 / 3, 16 / 3, 8 / 3])
    # 单组（groups=1）退化为 MHA：head0 配 K0、head1 配 K1，手算 sums=[2,4,6,4,4,3]
    q_mha = [[list(v) for v in Q4[0]], [list(v) for v in Q4[1]]]
    got1 = _verify(solve, [K0, K1], q_mha, 1, topk=6, kernel_size=3)
    assert np.allclose(got1["pooled"], [2.0, 4.0, 14 / 3, 14 / 3, 11 / 3, 7 / 3])


def check_last_query_only(solve):
    """只有最后一个 query 位置参与打分：改前面位置的 Q 不影响结果。"""
    base = _verify(solve, [K0, K1], Q4, 2, topk=3, kernel_size=3)
    qmod = [[list(v) for v in row] for row in Q4]
    qmod[0][0] = [99.0, -99.0]
    qmod[2][1] = [-50.0, 50.0]
    got = _verify(solve, [K0, K1], qmod, 2, topk=3, kernel_size=3)
    assert got == base


def check_pool_edges(solve):
    """池化两端零填充：分母恒为 kernel_size，边界窗口不含越界项。"""
    keys = [[[1, 0]] * 4]
    qs = [[[1, 0], [1, 0], [0, 1]]]
    got = _verify(solve, keys, qs, 1, topk=4, kernel_size=3)
    # q_last=(0,1) → sums=[0,0,0,0]；pooled 全 0 → indices 升序 0,1,2,3
    assert got["pooled"] == [0.0, 0.0, 0.0, 0.0]
    assert got["indices"] == [0, 1, 2, 3]

    keys2 = [[[2, 0], [0, 0], [0, 0], [0, 0]]]
    qs2 = [[[1, 0], [0, 0], [1, 0]]]
    got = _verify(solve, keys2, qs2, 1, topk=4, kernel_size=3)
    # q_last=(1,0) → sums=[2,0,0,0]；pooled = [2/3, 2/3, 0, 0]（j0 窗口只有 sums[0]、sums[1]）
    assert np.allclose(got["pooled"], [2 / 3, 2 / 3, 0.0, 0.0])
    assert got["indices"] == [0, 1, 2, 3]


def check_validation(solve):
    """形状、分组、topk/kernel/norm 取值等非法输入抛 ValueError。"""

    def bad(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("应抛 ValueError")

    good_k = [[[1, 0]], [[0, 1]]]
    good_q = [[[1, 0]], [[0, 1]], [[1, 1]], [[1, 0]]]
    bad(lambda: solve([], good_q, 2))
    bad(lambda: solve([[[]]], good_q, 2))
    bad(lambda: solve([[[1, 0]], [[0, 1], [1, 0]]], good_q, 2))
    bad(lambda: solve([[[1, 0]], [[0, 1], [1, 0]]], good_q, 2))
    bad(lambda: solve([[[1, 0]], [[0, 1]]], good_q, 1))          # 1*1 != 4 头
    bad(lambda: solve([[[1, 0]], [[0, 1]]], good_q, 3))          # 2*3 != 4
    bad(lambda: solve([[[1, 0]], [[0, 1]]], good_q, True))
    bad(lambda: solve([[[1, 0]], [[0, 1]]], good_q, 2, topk=0))
    bad(lambda: solve([[[1, 0]], [[0, 1]]], good_q, 2, topk=True))
    bad(lambda: solve([[[1, 0]], [[0, 1]]], good_q, 2, kernel_size=4))
    bad(lambda: solve([[[1, 0]], [[0, 1]]], good_q, 2, kernel_size=0))
    bad(lambda: solve([[[1, 0]], [[0, 1]]], good_q, 2, norm=0.0))
    bad(lambda: solve([[[1, 0]], [[0, 1]]], good_q, 2, norm=-1.0))
    bad(lambda: solve([[[1, 0]], [[0, float("inf")]]], good_q, 2))
    bad(lambda: solve([[[1, 0]], [[0, 1]]], good_q, 2, norm=float("nan")))
    bad(lambda: solve([[[1, 0]], [[0, 1]]], [], 2))
    bad(lambda: solve(good_k, [[[1, 0]], [[0, 1]], [[1, 1]]], 2))  # 头数 2*2 != 3
