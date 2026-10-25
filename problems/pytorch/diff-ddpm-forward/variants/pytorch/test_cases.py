import math

import torch

X0 = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
EPS = torch.tensor([[0.3, -0.7], [1.2, -0.4]])


def _oracle(x0, eps, ab):
    """独立重算：Python math.sqrt 标量系数逐元素乘加。"""
    sa = math.sqrt(ab)
    sb = math.sqrt(1 - ab)
    out = torch.empty_like(x0)
    for idx in range(x0.numel()):
        v0 = x0.flatten()[idx].item()
        ve = eps.flatten()[idx].item()
        out.flatten()[idx] = sa * v0 + sb * ve
    return out


def check_example(solve):
    y = solve(X0, EPS, 0.6)
    exp = torch.tensor([[0.9643333554267883, -1.9919122457504272],
                        [1.146245002746582, 2.070807933807373]])
    assert torch.allclose(y, exp, atol=1e-5, rtol=1e-5), f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(3)
    x0 = torch.randn(4, 5)
    eps = torch.randn(4, 5)
    for ab in [0.9, 0.5, 0.1]:
        assert torch.allclose(solve(x0, eps, ab), _oracle(x0, eps, ab), atol=1e-5, rtol=1e-5), ab


def check_alpha_one_is_x0(solve):
    y = solve(X0, EPS, 1.0)
    assert torch.allclose(y, X0, atol=1e-6, rtol=1e-6), "alpha_bar=1 时无噪声，x_t==x0"


def check_variance_coeffs(solve):
    # sqrt(ab)^2 + sqrt(1-ab)^2 的权重关系：用单位 x0、eps 探针系数
    ab = 0.35
    e = torch.ones(1)
    y = solve(torch.zeros(1), e, ab)
    assert abs(y.item() - math.sqrt(1 - ab)) < 1e-6, "x0=0 时系数应为 sqrt(1-ab)"


def check_highdim_shape(solve):
    x0 = torch.randn(2, 3, 4)
    eps = torch.randn(2, 3, 4)
    y = solve(x0, eps, 0.7)
    assert y.shape == (2, 3, 4) and y.dtype == torch.float32
