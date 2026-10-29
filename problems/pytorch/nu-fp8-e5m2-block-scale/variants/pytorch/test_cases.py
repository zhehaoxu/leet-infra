import bisect
import math

import torch

MAX_E5M2 = 57344.0


def _build_grid():
    """列出 E5M2 全部可表示幅值，并带上各自的位模式（ties-to-even 靠模式的奇偶判定）"""
    g = [(M * 2.0 ** -16, M) for M in range(4)]
    for E in range(1, 31):
        e = E - 15
        for M in range(4):
            g.append(((1.0 + M / 4.0) * 2.0 ** e, E * 4 + M))
    g.sort()
    return [v for v, _ in g], [p for _, p in g]


VALS, PAT = _build_grid()


def snap(t):
    """独立 oracle：在网格上做二分查找取最近，平局取尾数位为偶的一侧"""
    if math.isnan(t):
        return float('nan')
    a = abs(t)
    i = bisect.bisect_left(VALS, a)
    if i < len(VALS) and VALS[i] == a:
        c = a
    else:
        lo = VALS[i - 1] if i > 0 else None
        hi = VALS[i] if i < len(VALS) else None
        if lo is None:
            c = hi
        elif hi is None:
            c = lo
        elif a - lo < hi - a:
            c = lo
        elif hi - a < a - lo:
            c = hi
        else:
            c = lo if PAT[i - 1] % 2 == 0 else hi
    return -c if t < 0 else c


def scale_exp(m):
    """最小的整数 e 使 m <= 57344 * 2^e：纯循环搜索，不用 log2"""
    if m == 0.0:
        return 0
    e = -127
    while e < 127 and MAX_E5M2 * 2.0 ** e < m:
        e += 1
    return e


def oracle(x, block):
    flat = x.detach().to(torch.float64).reshape(-1).tolist()
    out = []
    for s in range(0, len(flat), block):
        seg = flat[s:s + block]
        m = max(abs(v) for v in seg)
        if not math.isfinite(m):
            out.extend([float('nan')] * len(seg))
            continue
        sc = 2.0 ** scale_exp(m)
        out.extend(snap(v / sc) * sc for v in seg)
    return torch.tensor(out, dtype=torch.float64).reshape(x.shape)


def _assert_same(got, exp):
    g = got.detach().to(torch.float64).reshape(-1).tolist()
    e = exp.reshape(-1).tolist()
    assert len(g) == len(e)
    for a, b in zip(g, e):
        if math.isnan(b):
            assert math.isnan(a), (a, b)
        else:
            assert a == b, (a, b)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    x = torch.tensor([1.0, 3.0, 5.0, 0.6, 0.5625])
    # absmax = 5 -> scale = 2^-13（5/57344 向上取到 2 的幂），块内乘 8192 后量化
    assert solve(x, block=5).tolist() == [1.0, 3.0, 5.0, 0.625, 0.5]
    # 切成两块：[1,3,5] 无损，[0.6,0.5625] 用更小的尺度，结果仍是同一组格点
    assert solve(x, block=3).tolist() == [1.0, 3.0, 5.0, 0.625, 0.5]


def check_matches_oracle(solve):
    torch.manual_seed(0)
    pool = [
        torch.randn(200) * 3.0,
        torch.randn(200) * 1e-8,
        torch.randn(200) * 1e8,
        torch.rand(200) * 1e-45,
        torch.rand(200) * 1e38,
        torch.tensor([0.6, 0.5625, 1e-5, 57344.0, -57344.0, 4608.0, 4915.2]),
    ]
    for x in pool:
        for block in (1, 3, 7, 32, 200):
            _assert_same(solve(x.float(), block), oracle(x.float(), block))


def check_scale_is_minimal_power_of_two(solve):
    # absmax 正好是 57344 * 2^k 时，尺度必须取 2^k：取小了会把最大值削掉一半
    for k in (-20, -5, 0, 1, 7, 30):
        m = MAX_E5M2 * 2.0 ** k
        x = torch.tensor([m, -m, m * 0.75, 1e-3])
        out = solve(x, block=4)
        assert float(out[0]) == m, (k, out.tolist())
        assert float(out[1]) == -m
        assert not torch.isinf(out).any()
    # 用非 2 的幂尺度（m/57344）会引入额外舍入：这里格点值必须原样保留
    y = torch.tensor([49152.0, 3.0, -1.5])
    _assert_same(solve(y, block=3), oracle(y, 3))


def check_ties_and_grid(solve):
    # 4608 介于 4096(M=0) 与 5120(M=1) 正中 -> 取偶 -> 4096
    x = torch.tensor([4608.0])
    assert float(solve(x, block=1)) == 4096.0
    # 4915.2 离 5120 更近
    assert float(solve(torch.tensor([4915.2]), block=1)) == 5120.0
    # 次正规区（步长 2^-16）的 ties：块尺度被 absmax=1.0 定成 2^-15，小元素落到 2^-16 网格
    # 1.5*2^-31 -> t = 1.5*2^-16 -> 平局取偶（M=2）-> 2^-16*2*2^-15 = 2^-30
    assert float(solve(torch.tensor([1.0, 1.5 * 2.0 ** -31]), block=2)[1]) == 2.0 ** -30
    # 2.5*2^-31 -> t = 2.5*2^-16 -> 平局取偶（M=2）
    assert float(solve(torch.tensor([1.0, 2.5 * 2.0 ** -31]), block=2)[1]) == 2.0 ** -30
    # 2^-33 -> t = 0.5*2^-16 -> 平局在 0(M=0, 偶) 与 2^-16(M=1) 之间 -> 取 0
    assert float(solve(torch.tensor([1.0, 2.0 ** -33]), block=2)[1]) == 0.0
    # 不在格点上的一般值：3.0*2^-20 距 3*2^-16 网格... 直接对拍 oracle
    y = torch.tensor([1.0, 3.0 * 2.0 ** -20, 1e-7, 0.1234])
    _assert_same(solve(y, block=4), oracle(y, 4))
    # 每个输出（除以块尺度后）都必须落在 E5M2 网格上
    torch.manual_seed(3)
    x = (torch.randn(64) * 100).float()
    out = solve(x, block=8)
    for s in range(0, 64, 8):
        sc = 2.0 ** scale_exp(float(x[s:s + 8].abs().max()))
        for v in out[s:s + 8].tolist():
            assert snap(v / sc) * sc == v, (v, sc)


def check_zeros_and_specials(solve):
    z = torch.tensor([0.0, -0.0, 0.0])
    out = solve(z, block=3)
    assert out.tolist() == [0.0, 0.0, 0.0]
    assert math.copysign(1.0, float(out[1])) > 0  # 舍入结果里的 0 统一为 +0.0
    for bad in (float('nan'), float('inf'), -float('inf')):
        x = torch.tensor([1.0, bad, 2.0, 3.0])
        o = solve(x, block=4)
        assert all(math.isnan(v) for v in o.tolist())
        # 只有同一块被污染，另一块照常
        o2 = solve(x, block=2)
        assert math.isnan(float(o2[0])) and not math.isnan(float(o2[3]))
    # 空张量
    assert solve(torch.zeros(0), block=4).numel() == 0
    # 形状与 dtype 保持一致
    m = torch.randn(2, 3, 5)
    assert solve(m, block=4).shape == m.shape
    assert solve(m, block=4).dtype == torch.float32
    assert solve(m.double(), block=4).dtype == torch.float64


def check_invalid(solve):
    x = torch.randn(4)
    for bad in (torch.tensor([1, 2, 3]), torch.ones(3, dtype=torch.bool), [1.0], None):
        try:
            solve(bad, 2)
            raise AssertionError('non-float input must raise')
        except ValueError:
            pass
    for block in (0, -1, 1.5, '2', None, True):
        try:
            solve(x, block)
            raise AssertionError('bad block must raise: %r' % (block,))
        except ValueError:
            pass


check_example.example = True
