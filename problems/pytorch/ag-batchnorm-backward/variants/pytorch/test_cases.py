import torch


def _bn_train_forward(x, gamma, beta, eps):
    """独立实现训练态 BN 前向（用同批统计量），供 autograd 产生参考梯度。"""
    mu = x.mean(dim=0, keepdim=True)
    var = (x - mu).pow(2).mean(dim=0, keepdim=True)
    return gamma * (x - mu) / (var + eps).sqrt() + beta


# 示例：两列都是 mu=0、有偏方差=1 ⇒ xhat=x（eps=0）
# gamma=[2,1]、dY=[[1,2],[3,4],[5,6],[7,8]] ⇒ a=[[2,2],[6,4],[10,6],[14,8]]
# col0: mean(a)=8、mean(a·xhat)=2 ⇒ dX = a-8-2·xhat = [-4,-4,4,4]
# col1: mean(a)=5、mean(a·xhat)=-1 ⇒ dX = a-5+xhat = [-2,-2,2,2]
EXAMPLE_X = [[-1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]]
EXAMPLE_GAMMA = [2.0, 1.0]
EXAMPLE_BETA = [0.0, 0.0]
EXAMPLE_DY = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
EXAMPLE_DX = [[-4.0, -2.0], [-4.0, -2.0], [4.0, 2.0], [4.0, 2.0]]
EXAMPLE_DGAMMA = [4.0, -4.0]
EXAMPLE_DBETA = [16.0, 20.0]


def check_example(solve):
    dX, dG, dB = solve(torch.tensor(EXAMPLE_X), torch.tensor(EXAMPLE_GAMMA),
                       torch.tensor(EXAMPLE_BETA), torch.tensor(EXAMPLE_DY), eps=0.0)
    assert torch.allclose(dX, torch.tensor(EXAMPLE_DX), atol=1e-6)
    assert torch.allclose(dG, torch.tensor(EXAMPLE_DGAMMA), atol=1e-6)
    assert torch.allclose(dB, torch.tensor(EXAMPLE_DBETA), atol=1e-6)


check_example.example = True


def check_matches_autograd(solve):
    torch.manual_seed(701)
    for n, c in ((8, 3), (5, 5), (64, 4)):
        x = torch.randn(n, c, dtype=torch.float64, requires_grad=True)
        gamma = torch.randn(c, dtype=torch.float64, requires_grad=True)
        beta = torch.randn(c, dtype=torch.float64, requires_grad=True)
        dY = torch.randn(n, c, dtype=torch.float64)
        y = _bn_train_forward(x, gamma, beta, 1e-5)
        refX, refG, refB = torch.autograd.grad((y * dY).sum(), [x, gamma, beta])
        gX, gG, gB = solve(x.detach(), gamma.detach(), beta.detach(), dY)
        assert torch.allclose(gX, refX, atol=1e-9), (n, c)
        assert torch.allclose(gG, refG, atol=1e-9), (n, c)
        assert torch.allclose(gB, refB, atol=1e-9), (n, c)


def check_two_projections(solve):
    # eps=0 时 BN 反向沿 batch 的两个投影精确为 0：Σ_N dX = 0、Σ_N dX·xhat = 0
    torch.manual_seed(702)
    x = torch.randn(9, 3, dtype=torch.float64)
    gamma = torch.randn(3, dtype=torch.float64)
    beta = torch.randn(3, dtype=torch.float64)
    dY = torch.randn(9, 3, dtype=torch.float64)
    dX, _, _ = solve(x, gamma, beta, dY, eps=0.0)
    assert torch.allclose(dX.sum(0), torch.zeros(3, dtype=torch.float64), atol=1e-9)
    xhat = (x - x.mean(0, keepdim=True)) / x.var(0, unbiased=False, keepdim=True).sqrt()
    assert torch.allclose((dX * xhat).sum(0), torch.zeros(3, dtype=torch.float64), atol=1e-8)


def check_batch_coupling_and_beta(solve):
    # 同一列内部互相耦合：把某一样本的 dY 置零会改变同列其它样本的 dX
    torch.manual_seed(703)
    x = torch.randn(4, 2, dtype=torch.float64)
    gamma = torch.ones(2, dtype=torch.float64)
    beta = torch.zeros(2, dtype=torch.float64)
    dY = torch.randn(4, 2, dtype=torch.float64)
    full = solve(x, gamma, beta, dY)[0]
    masked = dY.clone()
    masked[0] = 0.0
    part = solve(x, gamma, beta, masked)[0]
    assert (full[1:] - part[1:]).abs().max() > 1e-6
    assert torch.allclose(part.sum(0), torch.zeros(2, dtype=torch.float64), atol=1e-9)
    # beta 不影响 dX/dGamma
    b2 = solve(x, gamma, torch.randn(2, dtype=torch.float64), dY)
    ref = solve(x, gamma, beta, dY)
    assert torch.allclose(b2[0], ref[0], atol=1e-12)
    assert torch.allclose(b2[1], ref[1], atol=1e-12)
    assert torch.allclose(b2[2], ref[2], atol=1e-12)      # dBeta 同样与 beta 无关


def check_n1_degenerate_and_dtype(solve):
    x = torch.tensor([[1.0, 2.0]])
    gamma = torch.tensor([3.0, 4.0])
    beta = torch.tensor([1.0, 1.0])
    dY = torch.tensor([[5.0, 6.0]])
    dX, dG, dB = solve(x, gamma, beta, dY)
    assert dX.shape == (1, 2) and torch.isfinite(dX).all()
    assert torch.allclose(dX, torch.zeros(1, 2), atol=1e-6)   # 单样本：mu=x ⇒ 梯度全 0
    assert torch.allclose(dB, dY, atol=1e-6)
    dX64, _, _ = solve(x.double(), gamma.double(), beta.double(), dY.double())
    assert dX64.dtype == torch.float64
    assert torch.allclose(dX64.float(), dX, atol=1e-6)


def check_invalid_args(solve):
    x = torch.randn(4, 3)
    gamma = torch.ones(3)
    beta = torch.ones(3)
    dY = torch.randn(4, 3)
    bad = [
        (x.unsqueeze(0), gamma, beta, dY, 1e-5),   # x 非二维
        (x, torch.ones(2), beta, dY, 1e-5),        # gamma 长度不符
        (x, gamma, torch.ones(2), dY, 1e-5),       # beta 长度不符
        (x, gamma, beta, torch.randn(3, 4), 1e-5),  # dY 形状不符
        (x, gamma, beta, dY, -1.0),                # eps 负
    ]
    for xx, gg, bb, dy, ee in bad:
        try:
            solve(xx, gg, bb, dy, eps=ee)
        except ValueError:
            continue
        raise AssertionError("非法入参应抛 ValueError")
