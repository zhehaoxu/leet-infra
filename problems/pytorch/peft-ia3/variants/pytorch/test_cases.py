import torch

W = torch.tensor([[0.5, -0.2, 0.1, 0.3, 0.0],
                  [0.2, 0.4, -0.1, 0.0, 0.6],
                  [-0.3, 0.1, 0.5, 0.2, -0.4]])          # (3,5)
X = torch.tensor([[1.0, 0.5, -0.3, 0.2, 0.8],
                  [0.1, -0.4, 0.6, 0.3, -0.2]])            # (2,5)
LAM = torch.tensor([1.2, 0.8, 1.5])


def _oracle(x, W, lam):
    """独立重算：把 lam 折进权重行 W'=lam[:,None]*W，再一次矩阵乘（结构异于参考解的先乘后 scale）。"""
    Wp = lam.unsqueeze(1) * W
    return x @ Wp.T


def check_example(solve):
    y = solve(X, W, LAM)
    exp = torch.tensor([[0.5160000324249268, 0.7280000448226929, -1.0199999809265137],
                        [0.3360000550746918, -0.25600001215934753, 0.5550000667572021]])
    assert torch.allclose(y, exp, atol=1e-5, rtol=1e-5), f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(7)
    w = torch.randn(4, 6)
    x = torch.randn(5, 6)
    lam = torch.randn(4)
    y = solve(x, w, lam)
    assert torch.allclose(y, _oracle(x, w, lam), atol=1e-5, rtol=1e-5)


def check_unit_lambda_identity(solve):
    y = solve(X, W, torch.ones(3))
    assert torch.allclose(y, X @ W.T, atol=1e-6, rtol=1e-6), "lam=1 应等于普通线性"


def check_zero_lambda_kills_output(solve):
    y = solve(X, W, torch.zeros(3))
    assert torch.equal(y, torch.zeros_like(y)), "lam=0 应把对应输出维全部置零"


def check_3d_batch(solve):
    x3 = torch.stack([X, X.flip(0)])          # (2,2,5)
    y = solve(x3, W, LAM)
    assert y.shape == (2, 2, 3), f"3D 批量形状错: {tuple(y.shape)}"
    assert torch.allclose(y, _oracle(x3, W, LAM), atol=1e-5, rtol=1e-5)
