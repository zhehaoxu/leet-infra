import math
from decimal import Decimal, getcontext

import torch

getcontext().prec = 60

INF = float('inf')
NAN = float('nan')


def oracle(seg, p):
    """与缩放算法完全无关的 oracle：Decimal 高精度下直接 Σ|x|^p 再开 p 次方"""
    if len(seg) == 0:
        return 0.0
    if any(v != v for v in seg):
        return NAN
    if any(v == INF or v == -INF for v in seg):
        return INF
    m = max(abs(v) for v in seg)
    if m == 0.0:
        return 0.0
    if p == INF:
        return float(m)
    P = Decimal(p)
    total = sum(Decimal(abs(v)) ** P for v in seg)
    return float(total ** (Decimal(1) / P))


def expect(t, p, dim):
    """按输出顺序把每一段喂给 oracle（movedim 把被归约轴搬到最后一位）"""
    if dim is None:
        return [oracle(t.detach().to(torch.float64).reshape(-1).tolist(), p)]
    tt = torch.movedim(t.detach().to(torch.float64), dim, -1)
    n = tt.shape[-1]
    return [oracle(row, p) for row in tt.reshape(-1, n).tolist()]


def _flat(v):
    if isinstance(v, torch.Tensor):
        return v.detach().to(torch.float64).reshape(-1).tolist()
    out = []
    for item in v:
        if isinstance(item, (list, tuple)):
            out.extend(_flat(item))
        else:
            out.append(float(item))
    return out


def _close(got, want, rel):
    g = _flat(got)
    w = _flat(want)
    assert len(g) == len(w), (len(g), len(w))
    for a, b in zip(g, w):
        if math.isnan(b):
            assert math.isnan(a), (a, b)
        elif a == b:
            continue
        else:
            assert abs(a - b) <= rel * max(abs(a), abs(b)), (a, b)


def check_example(solve):
    x = torch.tensor([[3e154, 4e154, 0.0], [3e-200, 4e-200, 0.0]], dtype=torch.float64)
    _close(solve(x, 2.0, 1), [5e154, 5e-200], 1e-12)
    _close(solve(x, 1.0, 1), [7e154, 7e-200], 1e-12)
    _close(solve(x, INF, 1), [4e154, 4e-200], 0.0)
    # 不缩放直接平方：第一行上溢成 inf，第二行下溢成 0，torch 自带的 norm 也一起错
    naive = torch.sqrt((x * x).sum(dim=1))
    assert float(naive[0]) == INF and float(naive[1]) == 0.0
    builtin = torch.linalg.norm(x, dim=1)
    assert float(builtin[0]) == INF and float(builtin[1]) == 0.0
    # 可以手算的小数值：3-4-5 与 5-12-13
    _close(solve(torch.tensor([3.0, 4.0, 0.0]), 2.0), [5.0], 1e-12)
    _close(solve(torch.tensor([-5.0, 12.0]), 2.0), [13.0], 1e-12)
    _close(solve(torch.tensor([-5.0, 12.0]), 1.0), [17.0], 1e-12)
    _close(solve(torch.tensor([-5.0, 12.0]), INF), [12.0], 0.0)


def check_matches_oracle(solve):
    torch.manual_seed(0)
    base = torch.cat([
        torch.randn(4, 6, dtype=torch.float64) * 1e150,
        torch.randn(4, 6, dtype=torch.float64) * 1e-150,
        torch.randn(4, 6, dtype=torch.float64),
        torch.randn(4, 6, dtype=torch.float64).exp() * 1e-300,
    ], dim=0)
    base[0, 0] = 0.0
    base[3, 5] = 0.0
    for p in (1.0, 2.0, 3.0, 0.5, 8.0, INF):
        for dim in (None, 0, 1, -1):
            _close(solve(base, p, dim), expect(base, p, dim), 1e-9)
    # 量级差 1e300 混在一起时，小项被吃掉是正确行为而不是精度问题
    mixed = torch.tensor([[1e-300, 1.0, 3e30, 4e30]], dtype=torch.float64)
    _close(solve(mixed, 2.0, 1), expect(mixed, 2.0, 1), 1e-9)


def check_no_overflow_underflow(solve):
    # float32 的天花板是 3.4e38，2e38 平方直接爆，缩放后仍在 float32 范围内
    x32 = torch.tensor([2e38, 2e38], dtype=torch.float32)
    r32 = solve(x32, 2.0, 0)
    assert bool(torch.isfinite(r32)) and float(r32) > 0
    _close(r32, [oracle([2e38, 2e38], 2.0)], 1e-6)
    assert float(torch.linalg.norm(x32, dim=0)) == INF
    # 最小的 float64 次正规数：平方会彻底消失，开方后又必须回到 1 个 ulp
    tiny = torch.tensor([[5e-324, 5e-324]], dtype=torch.float64)
    rt = solve(tiny, 2.0, 1)
    assert float(rt[0]) > 0.0
    _close(rt, [oracle([5e-324, 5e-324], 2.0)], 1e-9)
    # 一万个 1e-2：缩放后每一项恰好是 1.0，求和不再受量级影响
    big = torch.full((10000,), 1e-2, dtype=torch.float64)
    _close(solve(big, 2.0, 0), expect(big.reshape(1, -1), 2.0, 1), 1e-9)
    # 超出 float32 范围的结果转成 inf 是允许的，但不能变成 NaN
    over = solve(torch.tensor([3e38, 3e38], dtype=torch.float32), 2.0, 0)
    assert float(over) == INF


def check_dims_keepdim_shape(solve):
    torch.manual_seed(1)
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    assert solve(x, 2.0, 0).shape == torch.Size([3, 4])
    assert solve(x, 2.0, 1).shape == torch.Size([2, 4])
    assert solve(x, 2.0, 2).shape == torch.Size([2, 3])
    assert solve(x, 2.0, -1).shape == torch.Size([2, 3])
    assert solve(x, 2.0, 0, True).shape == torch.Size([1, 3, 4])
    assert solve(x, 2.0, -1, True).shape == torch.Size([2, 3, 1])
    assert solve(x, 2.0).shape == torch.Size([])
    assert solve(x, 2.0, 2, True).shape == torch.Size([2, 3, 1])
    for dim in (0, 1, 2, -1, -3):
        _close(solve(x, 2.0, dim), expect(x, 2.0, dim), 1e-9)
    # dim 与 keepdim 的组合必须逐元素对齐原张量
    keep = solve(x, 2.0, 1, True)
    _close(keep, expect(x, 2.0, 1), 1e-9)
    # dtype 跟随输入
    f32 = torch.randn(2, 3)
    assert solve(f32, 2.0, 1).dtype == torch.float32
    assert solve(f32, 2.0).dtype == torch.float32
    assert solve(f32.double(), 2.0, 1).dtype == torch.float64
    # 单元素段：范数就是绝对值本身
    one = torch.tensor([[-3.0, 5.0]], dtype=torch.float64)
    _close(solve(one, 2.0, 0), [3.0, 5.0], 0.0)


def check_special_values(solve):
    # NaN 优先于 inf；inf 让整段变成 inf；整段是 0 时不做 0/0
    x = torch.tensor([
        [NAN, 1.0, 2.0],
        [INF, 1.0, 2.0],
        [-INF, NAN, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1e-320],
    ], dtype=torch.float64)
    for p in (1.0, 2.0, 8.0, INF):
        got = solve(x, p, 1)
        _close(got, expect(x, p, 1), 1e-9)
    assert math.isnan(float(solve(x, 2.0)))          # 整体归约里只要有一个 NaN 就是 NaN
    _close(solve(torch.zeros(2, 3), 2.0, 1), [0.0, 0.0], 0.0)
    _close(solve(torch.zeros(2, 3), 2.0), [0.0], 0.0)
    # 被归约轴长度为 0：每段都是空段，给 0.0
    e = torch.zeros(2, 0, 3, dtype=torch.float64)
    assert solve(e, 2.0, 1).shape == torch.Size([2, 3])
    _close(solve(e, 2.0, 1), torch.zeros(2, 3), 0.0)
    assert solve(e, 2.0, 1, True).shape == torch.Size([2, 1, 3])
    _close(solve(e, 3.0), [0.0], 0.0)
    assert solve(e, INF, 0).shape == torch.Size([0, 3])
    # 带符号的 0 不影响结果
    _close(solve(torch.tensor([-0.0, 0.0]), 2.0), [0.0], 0.0)


def check_invalid(solve):
    x = torch.randn(2, 3)
    for bad_p in (0.0, -1.0, -INF, NAN, True, '2', None, [2], torch.tensor(2.0)):
        try:
            solve(x, bad_p, 1)
            raise AssertionError('bad p must raise: %r' % (bad_p,))
        except ValueError:
            pass
    for bad_dim in (3, -4, 1.0, True, [1], '0'):
        try:
            solve(x, 2.0, bad_dim)
            raise AssertionError('bad dim must raise: %r' % (bad_dim,))
        except ValueError:
            pass
    for bad_keep in (1, 0, None, 'True'):
        try:
            solve(x, 2.0, 1, bad_keep)
            raise AssertionError('bad keepdim must raise: %r' % (bad_keep,))
        except ValueError:
            pass
    for bad_x in ([1.0, 2.0], 3, None, torch.ones(3, dtype=torch.int64),
                  torch.ones(3, dtype=torch.bool)):
        try:
            solve(bad_x, 2.0, 0)
            raise AssertionError('bad x must raise: %r' % (bad_x,))
        except ValueError:
            pass


check_example.example = True
