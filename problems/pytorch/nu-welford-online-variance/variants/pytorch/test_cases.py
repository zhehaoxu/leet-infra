import math

import torch


def d64(vals):
    return torch.tensor(vals, dtype=torch.float64)


def two_pass(vals, unbiased=True):
    """独立 oracle：先算均值再算二阶中心矩（两遍法，数值上足够精确）"""
    vals = list(vals)
    n = len(vals)
    m = math.fsum(vals) / n
    ss = math.fsum((v - m) ** 2 for v in vals)
    return m, ss / (n - 1) if unbiased else ss / n


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    mean, var = solve([torch.tensor([1.0, 2.0]), torch.tensor([]), torch.tensor([3.0, 4.0])])
    assert mean == 2.5
    assert abs(var - 5.0 / 3.0) < 1e-12
    mean, var = solve([torch.tensor([1.0, 2.0, 3.0, 4.0])], unbiased=False)
    assert mean == 2.5
    assert var == 1.25


def check_matches_two_pass(solve):
    torch.manual_seed(0)
    for n in (2, 33, 101, 1000):
        x = torch.randn(n)
        want_m, want_v = two_pass(x.tolist())
        got_m, got_v = solve([x])
        assert abs(got_m - want_m) <= 1e-9 * max(1.0, abs(want_m))
        assert abs(got_v - want_v) <= 1e-9 * max(1.0, abs(want_v))
        want_m, want_v = two_pass(x.tolist(), unbiased=False)
        got_m, got_v = solve([x], unbiased=False)
        assert abs(got_v - want_v) <= 1e-9 * max(1.0, abs(want_v))


def check_large_offset(solve):
    # 偏置 1e9、离散度 1：E[x^2]-E[x]^2 会把方差算成 0，两遍法和 Welford 给出 5/3
    xs = [1e9 + i for i in (1, 2, 3, 4)]
    mean, var = solve([d64(xs)])
    assert abs(mean - 1000000002.5) < 1e-9
    assert abs(var - 5.0 / 3.0) < 1e-9
    # 求和式写法在这里必然偏离（约 8.3e1 的误差量级）
    s = math.fsum(xs)
    sq = math.fsum(v * v for v in xs)
    naive_var = (sq - s * s / 4.0) / 3.0
    assert abs(naive_var - 5.0 / 3.0) > 1e-3
    big = [1e9 + float(i) for i in range(1000)]
    _, var = solve([d64(big[:400]), d64(big[400:])])
    want_m, want_v = two_pass(big)
    assert abs(want_v - 1000.0 * 1001.0 / 12.0) < 1e-6
    assert abs(var - want_v) <= 1e-9 * want_v


def check_chunking_invariant(solve):
    x = torch.arange(60.0)
    once = solve([x])
    many = solve([x[:7], x[7:20], x[20:21], x[21:], x[21:21]])
    assert once[0] == many[0] and once[1] == many[1]
    assert once == (29.5, 305.0)
    # 多块 + 二维块：按 row-major 展平后与一维等价
    m = torch.arange(24.0).reshape(4, 6)
    got = solve([m[:2].reshape(-1), m[2:]])
    want = solve([torch.arange(24.0)])
    assert got == want


def check_edges_and_specials(solve):
    for res in (solve([]), solve([torch.tensor([]), torch.tensor(())])):
        assert math.isnan(res[0]) and math.isnan(res[1])
    mean, var = solve([torch.tensor([3.5])])
    assert mean == 3.5 and math.isnan(var)
    mean, var = solve([torch.tensor([3.5])], unbiased=False)
    assert mean == 3.5 and var == 0.0
    for bad in ([torch.tensor([1.0, float('inf')])], [torch.tensor([1.0, float('nan')])],
                [torch.tensor([1.0]), torch.tensor([float('-inf')])]):
        mean, var = solve(bad)
        assert math.isnan(mean) and math.isnan(var)


def check_invalid(solve):
    for bad in ([torch.tensor([1, 2, 3])], [torch.tensor([1, 2], dtype=torch.int64)],
                [3.0], ['x'], torch.tensor([1.0]), 5):
        try:
            solve(bad)
            raise AssertionError('must raise: %r' % (bad,))
        except ValueError:
            pass
    for unb in (1, 0, None, 'true', 1.0):
        try:
            solve([torch.tensor([1.0, 2.0])], unb)
            raise AssertionError('unbiased must be bool')
        except ValueError:
            pass


check_example.example = True
