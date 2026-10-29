import math

import torch

INF = float('inf')
NAN = float('nan')
LN2 = 0.6931471805599453
LOG3 = 1.0986122886681098


def exact(x):
    """与分块算法无关的 oracle：整行一次性算，内部 float64"""
    src = x.detach().to(torch.float64)
    return torch.softmax(src, dim=1), torch.logsumexp(src, dim=1)


def _flat(v):
    """把 Tensor 或（可能嵌套的）list 拉平成一维 float 序列"""
    if isinstance(v, torch.Tensor):
        return v.detach().to(torch.float64).reshape(-1).tolist()
    out = []
    for item in v:
        if isinstance(item, (list, tuple)):
            out.extend(_flat(item))
        else:
            out.append(float(item))
    return out


def _close(got, want, tol):
    g = _flat(got)
    w = _flat(want)
    assert len(g) == len(w), (len(g), len(w))
    for a, b in zip(g, w):
        if math.isnan(b):
            assert math.isnan(a), (a, b)
        elif a == b:
            continue                        # inf 对 inf、-inf 对 -inf 直接放过
        else:
            assert abs(a - b) <= tol, (a, b)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    x = torch.tensor([[0.0, 0.0, 0.0], [0.0, LN2, -INF]], dtype=torch.float64)
    probs, lse = solve(x, 2)
    # exp 之后是 [1,1,1] 与 [1,2,0]，两行的分母都是 3
    _close(probs, [[1 / 3, 1 / 3, 1 / 3], [1 / 3, 2 / 3, 0.0]], 1e-12)
    _close(lse, [LOG3, LOG3], 1e-12)
    # 换一种切法（整行一段 / 每列一段），结果只能在最后一位上不同
    p1, l1 = solve(x, 1)
    p3, l3 = solve(x, 5)
    _close(p1, probs, 1e-12)
    _close(p3, probs, 1e-12)
    _close(l1, lse, 1e-12)


def check_matches_exact_for_any_shard(solve):
    torch.manual_seed(0)
    c = 7
    x = torch.cat([
        torch.randn(3, c, dtype=torch.float64) * 5.0,
        torch.randn(2, c, dtype=torch.float64) * 300.0,
        torch.randn(2, c, dtype=torch.float64).exp(),
        torch.full((1, c), -INF, dtype=torch.float64),
    ], dim=0)
    x[3, 2] = -INF
    x[4, 0] = 900.0
    x[5, c - 1] = -INF
    want_p, want_l = exact(x)
    for shard in range(1, c + 3):
        p, l = solve(x, shard)
        _close(p, want_p, 1e-9)
        _close(l, want_l, 1e-9)


def check_no_overflow_underflow(solve):
    x = torch.tensor([[900.0, -900.0, 1.0], [-900.0, -900.0, -800.0],
                      [1e3, 0.0, -1e3], [-1e3, -1e3, 2.0]], dtype=torch.float64)
    for shard in (1, 2, 3, 5):
        p, l = solve(x, shard)
        assert not bool(torch.isinf(p).any()) and not bool(torch.isnan(p).any())
        assert torch.all(p >= 0)
        _close(p.sum(dim=1), [1.0, 1.0, 1.0, 1.0], 1e-9)
        _close(l, [900.0, -800.0, 1000.0, 2.0], 1e-9)
    # 极端行里最大值的概率必须是 1（其余下溢成 0 是允许的）
    p, _ = solve(x, 2)
    assert float(p[0, 0]) == 1.0 and float(p[2, 0]) == 1.0
    # 朴素写法 exp(900) 会直接上溢，这道题因此要求参数恒 <= 0
    assert math.isinf(float(torch.exp(torch.tensor([900.0], dtype=torch.float64)[0])))


def check_huge_finite_max(solve):
    # 最大值 1e39 有限、但超出 float32 范围：把它误当 +inf 行会把 lse 变成 inf
    x = torch.tensor([[1e39, 0.0, -1e39], [-1e39, -1e39, 1e39]], dtype=torch.float64)
    for shard in (1, 2, 3):
        p, l = solve(x, shard)
        _close(p, [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], 0.0)
        assert bool(torch.isfinite(l).all()), l
        assert abs(float(l[0]) - 1e39) < 1e24 and abs(float(l[1]) - 1e39) < 1e24


def check_special_rows(solve):
    x = torch.tensor([
        [1.0, NAN, 3.0, 0.0],
        [INF, 1.0, 2.0, 3.0],
        [INF, INF, 0.0, -INF],
        [-INF, -INF, -INF, -INF],
        [-INF, 2.0, -INF, -INF],
    ], dtype=torch.float64)
    for shard in (1, 2, 3, 4):
        p, l = solve(x, shard)
        assert all(math.isnan(v) for v in p[0].tolist()) and math.isnan(float(l[0]))
        _close(p[1], [1.0, 0.0, 0.0, 0.0], 0.0)
        assert l[1].item() == INF
        _close(p[2], [0.5, 0.5, 0.0, 0.0], 1e-15)
        assert l[2].item() == INF
        assert all(math.isnan(v) for v in p[3].tolist()) and l[3].item() == -INF
        _close(p[4], [0.0, 1.0, 0.0, 0.0], 0.0)
        assert abs(float(l[4]) - 2.0) < 1e-12
    # 全 -inf 的段只是被跳过，不会把有有限值的行拖成 NaN
    y = torch.tensor([[-INF, -INF, 1.0]], dtype=torch.float64)
    p, l = solve(y, 2)
    _close(p, [[0.0, 0.0, 1.0]], 0.0)
    assert float(l[0]) == 1.0


def check_shapes_dtype_and_empty(solve):
    x = torch.randn(2, 5)
    p, l = solve(x, 3)
    assert p.shape == torch.Size([2, 5]) and p.dtype == torch.float32
    assert l.shape == torch.Size([2]) and l.dtype == torch.float32
    pd, ld = solve(x.double(), 3)
    assert pd.dtype == torch.float64 and ld.dtype == torch.float64
    # shard_size >= cols 时整行就是一段
    wp, wl = exact(x)
    _close(solve(x, 99)[0], wp, 1e-5)
    _close(solve(x, 99)[1], wl, 1e-5)
    e = torch.zeros(0, 4, dtype=torch.float64)
    pe, le = solve(e, 2)
    assert pe.shape == torch.Size([0, 4]) and le.shape == torch.Size([0])
    # 单列：有限行的概率是 1，全 -inf 行无定义
    p, l = solve(torch.tensor([[5.0], [-INF]], dtype=torch.float64), 1)
    _close(p, [[1.0], [NAN]], 0.0)
    _close(l, [5.0, -INF], 0.0)


def check_invalid(solve):
    x = torch.randn(2, 3)
    for bad in (torch.randn(2, 2, 2), torch.randn(3), torch.randn(1)):
        try:
            solve(bad, 2)
            raise AssertionError('must be 2-D')
        except ValueError:
            pass
    try:
        solve(torch.zeros(2, 0), 2)
        raise AssertionError('empty cols must raise')
    except ValueError:
        pass
    for bad in ([1.0, 2.0], 3, None, torch.ones(2, 3, dtype=torch.int32),
                torch.ones(2, 3, dtype=torch.bool)):
        try:
            solve(bad, 2)
            raise AssertionError('bad input must raise: %r' % (bad,))
        except ValueError:
            pass
    for shard in (0, -1, 1.5, '2', None, True):
        try:
            solve(x, shard)
            raise AssertionError('bad shard_size must raise: %r' % (shard,))
        except ValueError:
            pass


check_example.example = True
