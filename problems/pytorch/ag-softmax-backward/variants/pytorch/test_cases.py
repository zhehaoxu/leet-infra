import torch

# 示例：dim=-1，y 沿行和为 1，grad = [[1,2,3],[0,1,0]]
# 第 0 行 s = 0.5·1 + 0.25·2 + 0.25·3 = 1.75 ⇒ y·(grad − s) = [−0.375, 0.0625, 0.3125]
# 第 1 行 s = 0.5·1 = 0.5 ⇒ [−0.125, 0.25, −0.125]
EXAMPLE_Y = [[0.5, 0.25, 0.25], [0.25, 0.5, 0.25]]
EXAMPLE_GRAD = [[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]]
EXAMPLE_OUT = [[-0.375, 0.0625, 0.3125], [-0.125, 0.25, -0.125]]


def check_example(solve):
    out = solve(torch.tensor(EXAMPLE_Y), torch.tensor(EXAMPLE_GRAD))
    assert out.shape == (2, 3)
    assert torch.allclose(out, torch.tensor(EXAMPLE_OUT), atol=1e-6)


check_example.example = True


def check_matches_autograd_softmax(solve):
    torch.manual_seed(401)
    for dim in (-1, 0, 1):
        x = torch.randn(3, 4, 2, dtype=torch.float64, requires_grad=True)
        y = torch.softmax(x, dim=dim)
        g = torch.randn_like(y)
        (ref,) = torch.autograd.grad((y * g).sum(), x)
        got = solve(y.detach(), g, dim=dim)
        assert got.shape == x.shape and got.dtype == x.dtype
        assert torch.allclose(got, ref, atol=1e-9), dim


def check_log_softmax_path(solve):
    torch.manual_seed(402)
    x = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    y = torch.log_softmax(x, dim=-1)
    g = torch.randn_like(y)
    (ref,) = torch.autograd.grad((y * g).sum(), x)
    got = solve(y.detach(), g, log_input=True)
    assert torch.allclose(got, ref, atol=1e-9)
    # 两条路公式不同：同一个 y 换 log_input 结果必然不等（除平凡情形）
    plain = solve(y.detach().exp(), g, log_input=False)
    assert (got - plain).abs().max() > 1e-6


def check_middle_axis_and_negative(solve):
    torch.manual_seed(403)
    x = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    y = torch.softmax(x, dim=1)
    g = torch.randn_like(y)
    a = solve(y.detach(), g, dim=1)
    b = solve(y.detach(), g, dim=-2)
    assert torch.allclose(a, b, atol=1e-12)
    (ref,) = torch.autograd.grad((y * g).sum(), x)
    assert torch.allclose(a, ref, atol=1e-9)
    # 沿归一化轴求和后梯度为 0：Σ_i dx_i = 0
    assert torch.allclose(a.sum(dim=1), torch.zeros(2, 3, dtype=torch.float64), atol=1e-9)


def check_translation_invariance(solve):
    # 上游梯度沿 dim 为常数时，softmax 的反向必为 0（softmax 对 logits 平移不变）
    torch.manual_seed(404)
    x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    y = torch.softmax(x, dim=-1)
    for c in (1.0, -2.5, 7.0):
        got = solve(y.detach(), torch.full_like(y, c))
        assert torch.allclose(got, torch.zeros_like(got), atol=1e-12), c


def check_float32_and_1d(solve):
    y = torch.tensor([0.2, 0.8])
    g = torch.tensor([1.0, 1.0])
    out = solve(y, g)
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.zeros(2), atol=1e-7)
    big = torch.softmax(torch.randn(128, 512), dim=-1)
    gg = torch.randn_like(big)
    got = solve(big, gg)
    assert got.shape == big.shape and torch.isfinite(got).all()
    assert got.dtype == big.dtype


def check_invalid_args(solve):
    y = torch.softmax(torch.randn(3, 4), dim=-1)
    g = torch.randn(3, 4)
    bad = [
        (y, torch.randn(4, 3), -1, False),      # 形状不一致
        (y, torch.randn(3), -1, False),         # 少一维
        (y, g, 2, False),                        # dim 正数越界
        (y, g, -3, False),                       # dim 负数越界
        (y, g, -1, 1),                           # log_input 非 bool
        (y, g, -1, "yes"),                       # log_input 非 bool
    ]
    for yy, gg, dd, li in bad:
        try:
            solve(yy, gg, dim=dd, log_input=li)
        except ValueError:
            continue
        raise AssertionError("非法入参应抛 ValueError")
