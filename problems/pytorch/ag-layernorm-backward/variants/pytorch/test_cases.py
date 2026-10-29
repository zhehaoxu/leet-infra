import torch
import torch.nn.functional as F

# 示例：x = [-1,1,-1,1]（mu=0、有偏方差=1），eps=0 ⇒ xhat = x
# gamma=[2,1,2,1]、dY=[1,2,3,4] ⇒ g=[2,2,6,4]、mean(g)=3.5、mean(g·xhat)=-0.5
# dX = g - 3.5 + 0.5*xhat = [-2,-1,2,1]；dGamma = dY*xhat = [-1,2,-3,4]；dBeta = dY 之和
EXAMPLE_X = [[-1.0, 1.0, -1.0, 1.0]]
EXAMPLE_GAMMA = [2.0, 1.0, 2.0, 1.0]
EXAMPLE_BETA = [0.0, 0.0, 0.0, 0.0]
EXAMPLE_DY = [[1.0, 2.0, 3.0, 4.0]]
EXAMPLE_DX = [[-2.0, -1.0, 2.0, 1.0]]
EXAMPLE_DGAMMA = [-1.0, 2.0, -3.0, 4.0]
EXAMPLE_DBETA = [1.0, 2.0, 3.0, 4.0]


def check_example(solve):
    dX, dG, dB = solve(torch.tensor(EXAMPLE_X), torch.tensor(EXAMPLE_GAMMA),
                       torch.tensor(EXAMPLE_BETA), torch.tensor(EXAMPLE_DY), eps=0.0)
    assert torch.allclose(dX, torch.tensor(EXAMPLE_DX), atol=1e-6)
    assert torch.allclose(dG, torch.tensor(EXAMPLE_DGAMMA), atol=1e-6)
    assert torch.allclose(dB, torch.tensor(EXAMPLE_DBETA), atol=1e-6)


check_example.example = True


def check_matches_autograd(solve):
    torch.manual_seed(501)
    for shape in [(3, 8), (2, 5, 8), (8,)]:
        x = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        gamma = torch.randn(shape[-1], dtype=torch.float64, requires_grad=True)
        beta = torch.randn(shape[-1], dtype=torch.float64, requires_grad=True)
        dY = torch.randn_like(x.detach())
        y = F.layer_norm(x, (shape[-1],), gamma, beta, eps=1e-5)
        refX, refG, refB = torch.autograd.grad((y * dY).sum(), [x, gamma, beta])
        gX, gG, gB = solve(x.detach(), gamma.detach(), beta.detach(), dY)
        assert gX.shape == x.shape, shape
        assert torch.allclose(gX, refX, atol=1e-9), shape
        assert torch.allclose(gG, refG, atol=1e-9), shape
        assert torch.allclose(gB, refB, atol=1e-9), shape


def check_beta_unused_and_projections(solve):
    # β 只影响前向：换 β 梯度必须逐位不变
    torch.manual_seed(502)
    x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    gamma = torch.randn(6, dtype=torch.float64, requires_grad=True)
    dY = torch.randn_like(x.detach())
    a = solve(x.detach(), gamma.detach(), torch.zeros(6, dtype=torch.float64), dY, eps=0.0)
    b = solve(x.detach(), gamma.detach(), torch.randn(6, dtype=torch.float64), dY, eps=0.0)
    assert torch.allclose(a[0], b[0], atol=1e-12) and torch.allclose(a[1], b[1], atol=1e-12)
    # LN 反向的两个结构性约束：沿末轴 ΣdX = 0、ΣdX·xhat = 0
    dX = a[0]
    assert torch.allclose(dX.sum(-1), torch.zeros(4, dtype=torch.float64), atol=1e-9)
    xhat = (x.detach() - x.detach().mean(-1, keepdim=True)) / \
        x.detach().var(-1, unbiased=False, keepdim=True).sqrt()
    assert torch.allclose((dX * xhat).sum(-1), torch.zeros(4, dtype=torch.float64), atol=1e-9)


def check_identity_gamma_degenerates(solve):
    # gamma=1、beta=0 时 dGamma = Σ dY·xhat，dBeta = Σ dY；且与 softmax 无关的均值投影仍在
    torch.manual_seed(503)
    x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=torch.float64)
    ones = torch.ones(3, dtype=torch.float64)
    dY = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    dX, dG, dB = solve(x, ones, ones.new_zeros(3), dY, eps=0.0)
    assert dX.shape == (2, 3) and dG.shape == (3,) and dB.shape == (3,)
    assert torch.allclose(dB, dY.sum(0), atol=1e-12)
    std = (x - x.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True).sqrt()
    xhat = (x - x.mean(-1, keepdim=True)) / std
    assert torch.allclose(dG, (dY * xhat).sum(0), atol=1e-12)
    assert torch.isfinite(dX).all()


def check_dtype_and_1d(solve):
    x = torch.randn(6)
    gamma = torch.ones(6)
    beta = torch.zeros(6)
    dY = torch.randn(6)
    dX, dG, dB = solve(x, gamma, beta, dY)
    assert dX.dtype == torch.float32 and dX.shape == (6,)
    assert dG.shape == (6,) and dB.shape == (6,)
    dX64, _, _ = solve(x.double(), gamma.double(), beta.double(), dY.double())
    assert dX64.dtype == torch.float64
    assert torch.allclose(dX64.float(), dX, atol=1e-6)


def check_invalid_args(solve):
    x = torch.randn(2, 4)
    good = torch.ones(4)
    dY = torch.randn(2, 4)
    bad = [
        (x, torch.ones(3), good, dY),      # gamma 长度与末轴不符
        (x, good, torch.ones(3), dY),      # beta 长度不符
        (x, good.reshape(2, 2), good, dY),  # gamma 非一维
        (x, good, good, torch.randn(4, 2)),  # dY 形状不符
    ]
    for xx, gg, bb, dy in bad:
        try:
            solve(xx, gg, bb, dy)
        except ValueError:
            continue
        raise AssertionError("非法入参应抛 ValueError")
