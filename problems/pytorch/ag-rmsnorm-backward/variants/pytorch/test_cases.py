import torch


def _rms_forward(x, gamma, eps):
    """独立写一遍前向（不借 F.rms_norm），供 autograd 求参考梯度。"""
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    return gamma * (x / (ms + eps).sqrt())


# 示例：x = [-1,1,-1,1] ⇒ mean(x²)=1、rms=1、xhat=x；gamma=[2,1,2,1]、dY=[1,2,3,4]
# g=[2,2,6,4]、mean(g·xhat) = -0.5 ⇒ dX = g + 0.5·xhat = [1.5, 2.5, 5.5, 4.5]
EXAMPLE_X = [[-1.0, 1.0, -1.0, 1.0]]
EXAMPLE_GAMMA = [2.0, 1.0, 2.0, 1.0]
EXAMPLE_DY = [[1.0, 2.0, 3.0, 4.0]]
EXAMPLE_DX = [[1.5, 2.5, 5.5, 4.5]]
EXAMPLE_DGAMMA = [-1.0, 2.0, -3.0, 4.0]


def check_example(solve):
    dX, dG = solve(torch.tensor(EXAMPLE_X), torch.tensor(EXAMPLE_GAMMA),
                   torch.tensor(EXAMPLE_DY), eps=0.0)
    assert torch.allclose(dX, torch.tensor(EXAMPLE_DX), atol=1e-6)
    assert torch.allclose(dG, torch.tensor(EXAMPLE_DGAMMA), atol=1e-6)


check_example.example = True


def check_matches_autograd(solve):
    torch.manual_seed(601)
    for shape in [(3, 8), (2, 4, 6), (5,)]:
        x = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        gamma = torch.randn(shape[-1], dtype=torch.float64, requires_grad=True)
        dY = torch.randn(*shape, dtype=torch.float64)
        y = _rms_forward(x, gamma, 1e-6)
        refX, refG = torch.autograd.grad((y * dY).sum(), [x, gamma])
        gX, gG = solve(x.detach(), gamma.detach(), dY, eps=1e-6)
        assert gX.shape == x.shape and gG.shape == (shape[-1],), shape
        assert torch.allclose(gX, refX, atol=1e-9), shape
        assert torch.allclose(gG, refG, atol=1e-9), shape


def check_not_the_layernorm_formula(solve):
    # dY 沿末轴为常数时：RMSNorm 的 dX 非 0（LayerNorm 版公式会错误地给 0）
    torch.manual_seed(602)
    x = torch.randn(2, 5, dtype=torch.float64)
    gamma = torch.ones(5, dtype=torch.float64)
    dY = torch.full((2, 5), 3.0, dtype=torch.float64)
    dX, _ = solve(x, gamma, dY, eps=0.0)
    assert dX.abs().max() > 1e-3
    rms = x.pow(2).mean(-1, keepdim=True).sqrt()
    assert torch.allclose(dX, dY / rms - x * (dY * x).mean(-1, keepdim=True)
                          / rms.pow(3), atol=1e-9)


def check_gamma_beta_fold(solve):
    # dGamma = Σ_前导轴 dY·xhat，逐 (b, s) 相加与整体折叠等价
    torch.manual_seed(603)
    x = torch.randn(3, 4, 6, dtype=torch.float64)
    gamma = torch.randn(6, dtype=torch.float64)
    dY = torch.randn(3, 4, 6, dtype=torch.float64)
    rms = x.pow(2).mean(-1, keepdim=True).sqrt()
    xhat = x / rms
    ref = (dY * xhat).sum(dim=(0, 1))
    _, got = solve(x, gamma, dY, eps=0.0)
    assert torch.allclose(got, ref, atol=1e-9)
    # gamma 全 1 时 dX 与 gamma 无关的那条缩放：dX(gamma=1) = (dY − xhat·mean(dY·xhat))/rms
    _, g1 = solve(x, torch.ones(6, dtype=torch.float64), dY, eps=0.0)
    assert torch.allclose(g1, ref, atol=1e-9)


def check_degenerate_and_dtype(solve):
    z = torch.zeros(1, 4)
    gamma = torch.ones(4)
    dY = torch.ones(1, 4)
    dX, dG = solve(z, gamma, dY, eps=1e-6)
    assert torch.isfinite(dX).all() and torch.isfinite(dG).all()
    assert dX.dtype == torch.float32 and dG.dtype == torch.float32
    dX64, dG64 = solve(z.double(), gamma.double(), dY.double(), eps=1e-6)
    assert dX64.dtype == torch.float64 and dG64.dtype == torch.float64
    assert torch.allclose(dX64.float(), dX, atol=1e-6)


def check_invalid_args(solve):
    x = torch.randn(2, 4)
    gamma = torch.ones(4)
    dY = torch.randn(2, 4)
    bad = [
        (x, torch.ones(3), dY, 1e-6),       # gamma 长度不符
        (x, gamma.reshape(2, 2), dY, 1e-6),  # gamma 非一维
        (x, gamma, torch.randn(4, 2), 1e-6), # dY 形状不符
        (x, gamma, dY, -1e-3),               # eps 负
    ]
    for xx, gg, dy, ee in bad:
        try:
            solve(xx, gg, dy, eps=ee)
        except ValueError:
            continue
        raise AssertionError("非法入参应抛 ValueError")
