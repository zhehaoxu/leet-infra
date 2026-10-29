import torch
import torch.nn.functional as F

# 示例：relu 在 x<=0 处梯度取 0（含 x 恰好为 0），dY 全 1
EXAMPLE_X = [-2.0, -0.5, 0.0, 0.5, 2.0]
EXAMPLE_DY = [1.0, 1.0, 1.0, 1.0, 1.0]
EXAMPLE_OUT = [0.0, 0.0, 0.0, 1.0, 1.0]


def _t(v):
    return torch.tensor(v, dtype=torch.float64)


def check_example(solve):
    out = solve(_t(EXAMPLE_X), _t(EXAMPLE_DY), 'relu')
    assert torch.allclose(out, _t(EXAMPLE_OUT), atol=1e-9)


check_example.example = True


def _fwd(kind, x, ns):
    if kind == 'relu':
        return F.relu(x)
    if kind == 'leaky_relu':
        return F.leaky_relu(x, negative_slope=ns)
    if kind == 'elu':
        return F.elu(x)
    if kind == 'sigmoid':
        return torch.sigmoid(x)
    if kind == 'tanh':
        return torch.tanh(x)
    if kind == 'gelu':
        return F.gelu(x)
    if kind == 'silu':
        return F.silu(x)
    return F.softplus(x)


def check_matches_autograd_all_kinds(solve):
    torch.manual_seed(1301)
    for kind in ('relu', 'leaky_relu', 'elu', 'sigmoid', 'tanh', 'gelu', 'silu', 'softplus'):
        x = torch.randn(40, dtype=torch.float64, requires_grad=True)
        dY = torch.randn(40, dtype=torch.float64)
        (ref,) = torch.autograd.grad((_fwd(kind, x, 0.2) * dY).sum(), x)
        got = solve(x.detach(), dY, kind, negative_slope=0.2)
        assert torch.allclose(got, ref, atol=1e-9), (kind, (got - ref).abs().max())


def check_zero_point_conventions(solve):
    x = _t([0.0])
    one = _t([1.0])
    assert float(solve(x, one, 'relu')) == 0.0                 # torch 在 0 处给 0
    assert float(solve(x, one, 'leaky_relu', negative_slope=0.1)) == 0.1
    assert float(solve(x, one, 'elu')) == 1.0
    assert abs(float(solve(x, one, 'sigmoid')) - 0.25) < 1e-12
    assert abs(float(solve(x, one, 'tanh')) - 1.0) < 1e-12
    assert abs(float(solve(x, one, 'gelu')) - 0.5) < 1e-12
    assert abs(float(solve(x, one, 'silu')) - 0.5) < 1e-12
    assert abs(float(solve(x, one, 'softplus')) - 0.5) < 1e-12


def check_gelu_is_exact_variant(solve):
    # 精确 gelu 与 tanh 近似在 |x|≈1 附近有明显差别，用它排除近似式
    torch.manual_seed(1302)
    x = _t([-1.35, -1.0, 1.0, 1.35])
    one = torch.ones_like(x)
    got = solve(x, one, 'gelu')
    approx = 0.5 * (1 + x.tanh())  # 故意写错的对照，必须与解析式不同
    assert (got - approx).abs().max() > 1e-3
    xr = x.clone().requires_grad_(True)
    (ref,) = torch.autograd.grad((F.gelu(xr) * one).sum(), xr)
    assert torch.allclose(got, ref, atol=1e-12)


def check_shapes_negative_slope_scope_and_dtype(solve):
    x = torch.randn(3, 4)
    dY = torch.randn(3, 4)
    assert solve(x, dY, 'relu').dtype == torch.float32
    assert solve(x.double(), dY.double(), 'relu').dtype == torch.float64
    # negative_slope 只影响 leaky_relu
    a = solve(x, dY, 'relu')
    b = solve(x, dY, 'relu', negative_slope=0.7)
    assert torch.allclose(a, b, atol=0.0)
    # 2D/4D 都按逐元素处理，且 dx 与 dY 同号当导数恒正
    xg = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    dg = torch.randn_like(xg)
    assert solve(xg, dg, 'softplus').shape == xg.shape
    pos = torch.zeros(6, dtype=torch.float64)
    assert torch.allclose(solve(pos, _t([1.0, -1.0, 2.0, 0.0, 3.0, 1.0]), 'tanh'),
                          _t([1.0, -1.0, 2.0, 0.0, 3.0, 1.0]), atol=1e-12)


def check_invalid_args(solve):
    x = torch.randn(4)
    dY = torch.randn(4)
    bad = [
        (x, torch.randn(5), 'relu', 0.01),     # 形状不符
        (x, dY, 'prelu', 0.01),                # 未知 kind
        (x, dY, 'RELU2', 0.01),                # 未知 kind
        (x, dY, 5, 0.01),                      # kind 非字符串
    ]
    for xx, dy, kk, ns in bad:
        try:
            solve(xx, dy, kk, negative_slope=ns)
        except ValueError:
            continue
        raise AssertionError("非法入参应抛 ValueError")
