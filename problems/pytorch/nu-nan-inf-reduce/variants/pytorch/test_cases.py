import math

import torch

NAN = float('nan')
INF = float('inf')
MODES = ('sum', 'mean', 'max', 'min')


def _segs(x, dim):
    """把 x 切成一段段待归约的序列（纯 Python），同时给出不带 keepdim 的输出形状"""
    if dim is None:
        return [x.reshape(-1).double().tolist()], []
    d = dim % x.dim()
    y = x.movedim(d, -1).reshape(-1, x.shape[d])
    shape = list(x.shape[:d]) + list(x.shape[d + 1:])
    return [r.tolist() for r in y.double()], shape


def _reduce(seg, mode):
    v = [a for a in seg if not math.isnan(a)]
    if mode == 'sum':
        return sum(v)
    if not v:
        return NAN
    if mode == 'mean':
        return sum(v) / len(v)
    return max(v) if mode == 'max' else min(v)


def _oracle(x, dim, mode):
    segs, shape = _segs(x, dim)
    vals = [_reduce(s, mode) for s in segs]
    if dim is None:
        return torch.tensor(vals[0], dtype=torch.float64)
    return torch.tensor(vals, dtype=torch.float64).reshape(shape)


def _eq(got, exp):
    """逐元素比较，NaN 位置必须也是 NaN"""
    got = list(got)
    assert len(got) == len(exp), (got, exp)
    for a, b in zip(got, exp):
        if math.isnan(b):
            assert math.isnan(a), (a, b)
        else:
            assert a == b, (a, b)


def _assert_close(got, exp, tol):
    g = got.double().reshape(-1).tolist()
    e = exp.double().reshape(-1).tolist()
    assert len(g) == len(e), (got.shape, exp.shape)
    for a, b in zip(g, e):
        if math.isnan(a) or math.isnan(b):
            assert math.isnan(a) and math.isnan(b), (a, b)
        elif a == b:
            continue  # inf 对 inf：减法会得到 nan，必须先放行
        else:
            assert abs(a - b) <= tol * max(1.0, abs(b)), (a, b)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    x = torch.tensor([[1.0, NAN, 3.0], [NAN, NAN, 2.0], [NAN, NAN, NAN]])
    _eq(solve(x, 1, 'mean').tolist(), [2.0, 2.0, NAN])
    _eq(solve(x, 1, 'sum').tolist(), [4.0, 2.0, 0.0])
    _eq(solve(x, 1, 'max').tolist(), [3.0, 2.0, NAN])
    _eq(solve(x, 0, 'mean').tolist(), [1.0, NAN, 2.5])
    _eq(solve(x, 0, 'sum').tolist(), [1.0, 0.0, 5.0])
    assert float(solve(x, None, 'sum')) == 6.0
    assert float(solve(x, None, 'mean')) == 2.0
    assert float(solve(x, None, 'max')) == 3.0
    assert float(solve(x, None, 'min')) == 1.0


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for shape in ((6,), (4, 5), (2, 3, 4)):
        x = torch.randn(*shape, dtype=torch.float64)
        mask = torch.rand(*shape, dtype=torch.float64) < 0.35
        x = torch.where(mask, torch.full_like(x, NAN), x)
        for mode in MODES:
            for dim in list(range(-x.dim(), x.dim())) + [None]:
                exp = _oracle(x, dim, mode)
                _assert_close(solve(x, dim, mode), exp, 1e-12)
                _assert_close(solve(x, dim, mode, keepdim=True), exp, 1e-12)


def check_all_nan_segment(solve):
    x = torch.tensor([[NAN, 1.0, NAN], [NAN, NAN, NAN]])
    _eq(solve(x, 1, 'sum').tolist(), [1.0, 0.0])
    for mode in ('mean', 'max', 'min'):
        _eq(solve(x, 1, mode).tolist(), [1.0, NAN])
    _eq(solve(x[1:], 1, 'sum').tolist(), [0.0])
    _eq(solve(x[1:], 1, 'max').tolist(), [NAN])
    _eq(solve(x, 0, 'sum').tolist(), [0.0, 1.0, 0.0])
    _eq(solve(x, 0, 'mean').tolist(), [NAN, 1.0, NAN])
    # 分母必须是有效元素个数：若误用 numel()，第一行会给出 1/3
    assert abs(float(solve(x, 0, 'mean')[1]) - 1.0) < 1e-12


def check_inf_participates(solve):
    x = torch.tensor([[INF, NAN, -INF], [2.0, NAN, 3.0]])
    _eq(solve(x, 1, 'max').tolist(), [INF, 3.0])
    _eq(solve(x, 1, 'min').tolist(), [-INF, 2.0])
    _eq(solve(x, 1, 'sum').tolist(), [NAN, 5.0])
    _eq(solve(x, 0, 'max').tolist(), [INF, NAN, 3.0])
    y = torch.tensor([[-INF, 1.0], [INF, NAN]])
    for mode in MODES:
        _assert_close(solve(y, 1, mode), _oracle(y, 1, mode), 1e-12)


def check_shapes_dtypes_keepdim(solve):
    x = torch.randn(2, 3, 4)
    for d in range(3):
        exp_shape = (2, 3, 4)[:d] + (2, 3, 4)[d + 1:]
        assert solve(x, d, 'mean').shape == torch.Size(exp_shape)
        assert solve(x, d - 3, 'mean').shape == torch.Size(exp_shape)
        assert solve(x, d, 'mean', keepdim=True).shape == \
            torch.Size((2, 3, 4)[:d] + (1,) + (2, 3, 4)[d + 1:])
    assert solve(x, None, 'mean').dim() == 0
    assert solve(x, 1, 'mean').dtype == torch.float32
    assert solve(x.double(), 1, 'mean').dtype == torch.float64
    y = torch.tensor([[1.5, NAN, -2.25]], dtype=torch.float32)
    _assert_close(solve(y, 1, 'mean'), _oracle(y, 1, 'mean'), 1e-6)
    # 归约轴长度为 0：段里没有任何元素，等价于 count == 0
    e = torch.zeros(0, 3)
    assert solve(e, 0, 'sum').shape == torch.Size([3])
    assert solve(e, 0, 'mean').shape == torch.Size([3])
    assert solve(e, 0, 'max', keepdim=True).shape == torch.Size([1, 3])
    for mode in MODES:
        got = solve(e, 0, mode).tolist()
        want = [0.0] * 3 if mode == 'sum' else [NAN] * 3
        _eq(got, want)


def check_invalid(solve):
    x = torch.randn(2, 3)
    for bad in (torch.tensor([1, 2]), torch.ones(2, dtype=torch.bool), [1.0], 3):
        try:
            solve(bad, 0, 'mean')
            raise AssertionError('non-float input must raise')
        except ValueError:
            pass
    for mode in ('prod', 'MEAN', '', None, 0):
        try:
            solve(x, 1, mode)
            raise AssertionError('bad mode must raise')
        except ValueError:
            pass
    for dim in (2, -3, 1.0):
        try:
            solve(x, dim, 'mean')
            raise AssertionError('bad dim must raise: %r' % (dim,))
        except ValueError:
            pass
    for keepdim in (0, 1, None):
        try:
            solve(x, 1, 'mean', keepdim)
            raise AssertionError('bad keepdim must raise')
        except ValueError:
            pass


check_example.example = True
