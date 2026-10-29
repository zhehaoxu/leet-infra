import math
import struct

import torch

VARIANTS = ('sr', 'fwd', 'bwd')
MAX_BF16 = 3.3895313892515355e+38


def _pat(v):
    return struct.unpack('<I', struct.pack('<f', v))[0]


def _f32(p):
    return struct.unpack('<f', struct.pack('<I', p & 0xFFFFFFFF))[0]


def p_of(r, variant):
    if variant == 'sr':
        return r
    if variant == 'fwd':
        return 0.0 if r <= 0.5 else 2.0 * r - 1.0
    return 1.0 if r >= 0.5 else 2.0 * r


def oracle(x, u, variant):
    """独立实现：用 struct 在 Python 侧走一遍位运算，不碰 torch 的张量视图"""
    out = []
    for v, n in zip(x.reshape(-1).tolist(), u.reshape(-1).tolist()):
        pat = _pat(v)
        r = (pat & 0xFFFF) / 65536.0
        hi = (pat >> 16) + 1
        pick = hi if n < p_of(r, variant) else (pat >> 16)
        out.append(_f32(pick << 16))
    return torch.tensor(out, dtype=torch.float32).reshape(x.shape)


def _assert_bitwise(got, exp):
    g = _pat_list(got)
    e = _pat_list(exp)
    assert len(g) == len(e), (g, e)
    for a, b in zip(g, e):
        assert a == b, (hex(a), hex(b))


def _pat_list(t):
    return [_pat(v) for v in t.reshape(-1).tolist()]


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    # bf16 在 [1,2) 的 ulp 是 2^-7：1.001953125 = 1 + 2^-9 -> r = 0.25
    #                              1.005859375 = 1 + 3*2^-9 -> r = 0.75
    x = torch.tensor([1.001953125, 1.005859375])
    u = torch.tensor([0.25, 0.5])
    assert solve(x, u, 'sr').tolist() == [1.0, 1.0078125]
    assert solve(x, u, 'fwd').tolist() == [1.0, 1.0]
    assert solve(x, u, 'bwd').tolist() == [1.0078125, 1.0078125]
    u2 = torch.tensor([0.0, 0.9])
    assert solve(x, u2, 'sr').tolist() == [1.0078125, 1.0]
    assert solve(x, u2, 'fwd').tolist() == [1.0, 1.0]
    assert solve(x, u2, 'bwd').tolist() == [1.0078125, 1.0078125]


def check_matches_oracle(solve):
    for base in (1.0, -1.0, 3.0, 0.0078125, -1e-4, 1e10, -2.0 ** -130):
        # 同一区间内取满 8 个残差，覆盖 r = 0, 1/8, ... 7/8
        k = torch.arange(8).float()
        x = (torch.tensor([base]).abs().item() * 0 + base) + k * (abs(base) * 2.0 ** -10)
        x = torch.cat([x, -x])
        for un in (0.0, 0.125, 0.5, 0.875, 0.9999):
            u = torch.full_like(x, un)
            for variant in VARIANTS:
                _assert_bitwise(solve(x, u, variant), oracle(x, u, variant))
    torch.manual_seed(7)
    xr = (torch.randn(300) * 10).float()
    ur = torch.rand(300)
    for variant in VARIANTS:
        _assert_bitwise(solve(xr, ur, variant), oracle(xr, ur, variant))


def check_bias_directions(solve):
    torch.manual_seed(0)
    n = 40000
    x = (torch.rand(n) * 1.9 + 1.0).float()
    u = torch.rand(n)
    want = x.to(torch.float64)
    sr = solve(x, u, 'sr').to(torch.float64)
    fwd = solve(x, u, 'fwd').to(torch.float64)
    bwd = solve(x, u, 'bwd').to(torch.float64)
    assert float((sr - want).mean()).__abs__() < 2e-4
    assert float((fwd - want).mean()) < -1e-3   # 系统性偏向零
    assert float((bwd - want).mean()) > 1e-3    # 系统性远离零
    assert float(fwd.mean()) < float(sr.mean()) < float(bwd.mean())
    # 朴素截断（RZ）比 fwd 还要下偏，SR 的无偏性是它的卖点
    rz = torch.where(x >= 0, torch.floor(x * 128.0) / 128.0, torch.ceil(x * 128.0) / 128.0)
    assert float((rz.to(torch.float64) - want).mean()) < float((fwd - want).mean())


def check_ties_and_strict_compare(solve):
    # r = 0.5 的格点中点：u = 0.5 时 'sr'/'fwd' 都不进位（严格小于），'bwd' 必进位（p = 1）
    x = torch.tensor([1.00390625, -1.00390625])
    u = torch.tensor([0.5, 0.5])
    assert solve(x, u, 'sr').tolist() == [1.0, -1.0]
    assert solve(x, u, 'fwd').tolist() == [1.0, -1.0]
    assert solve(x, u, 'bwd').tolist() == [1.0078125, -1.0078125]
    # u 恰等于 p 时一律不进位（r = 0.25 -> p = 0.25）
    assert solve(torch.tensor([1.001953125]), torch.tensor([0.25]), 'sr').tolist() == [1.0]
    assert solve(torch.tensor([1.001953125]), torch.tensor([0.24999]), 'sr').tolist() == [1.0078125]
    # r = 0.75 时 'sr' 的 p = 0.75，u = 0.75 正好卡在边界上不进位
    assert solve(torch.tensor([1.005859375]), torch.tensor([0.75]), 'sr').tolist() == [1.0]
    assert solve(torch.tensor([1.005859375]), torch.tensor([0.74999]), 'sr').tolist() == [1.0078125]


def check_exact_grid_and_specials(solve):
    grid = torch.tensor([1.0, 1.0078125, -2.5, 0.5, 65536.0, 2.0 ** -133, 0.0, -0.0])
    for variant in VARIANTS:
        for un in (0.0, 0.5, 0.999999):
            u = torch.full_like(grid, un)
            got = solve(grid, u, variant)
            _assert_bitwise(got, grid)
        # ±0 的符号必须保留
        z = solve(torch.tensor([0.0, -0.0]), torch.tensor([0.3, 0.3]), variant)
        assert math.copysign(1.0, float(z[0])) > 0
        assert math.copysign(1.0, float(z[1])) < 0
    i = torch.tensor([float('inf'), float('-inf')])
    for variant in VARIANTS:
        assert solve(i, torch.full_like(i, 0.5), variant).tolist() == [float('inf'), float('-inf')]
    nn = torch.tensor([float('nan'), float('nan')])
    for variant in VARIANTS:
        assert all(math.isnan(v) for v in solve(nn, torch.full_like(nn, 0.5), variant).tolist())
    # 超过 bf16 最大有限值：进位溢出成 inf，舍去落在最大有限值
    big = torch.tensor([3.4e38, 3.4e38])
    assert math.isinf(float(solve(big, torch.tensor([0.0, 0.99]), 'sr')[0]))
    assert float(solve(big, torch.tensor([0.0, 0.99]), 'sr')[1]) == MAX_BF16
    assert float(solve(big, torch.tensor([0.99, 0.99]), 'sr')[0]) == MAX_BF16


def check_invalid(solve):
    x = torch.randn(5)
    u = torch.rand(5)
    for bad in (x.double(), x.half(), torch.tensor([1, 2, 3, 4, 5])):
        try:
            solve(bad, u, 'sr')
            raise AssertionError('x dtype must raise')
        except ValueError:
            pass
    try:
        solve(x, u.double(), 'sr')
        raise AssertionError('u dtype must raise')
    except ValueError:
        pass
    for bad_u in (torch.full((5,), 1.0), torch.full((5,), -0.5), torch.tensor([0.1] * 4 + [float('nan')]),
                  torch.full((5,), float('inf'))):
        try:
            solve(x, bad_u, 'sr')
            raise AssertionError('u range must raise')
        except ValueError:
            pass
    for bad_shape in (torch.rand(4), torch.rand(5, 1)):
        try:
            solve(x, bad_shape, 'sr')
            raise AssertionError('shape mismatch must raise')
        except ValueError:
            pass
    for variant in ('SR', 'stochastic', '', None, 0):
        try:
            solve(x, u, variant)
            raise AssertionError('bad variant must raise')
        except ValueError:
            pass
    try:
        solve([1.0], u, 'sr')
        raise AssertionError('non-tensor must raise')
    except ValueError:
        pass
    # 默认 variant 是 sr，形状保持
    m = torch.randn(2, 3)
    assert solve(m, torch.rand(2, 3)).shape == torch.Size([2, 3])
    assert torch.equal(solve(m, torch.zeros(2, 3)), solve(m, torch.zeros(2, 3), 'sr'))


check_example.example = True
