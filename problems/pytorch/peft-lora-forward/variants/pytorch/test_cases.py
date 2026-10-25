import torch

W0 = torch.tensor([[0.5, -0.2, 0.1, 0.3, 0.0],
                   [0.2, 0.4, -0.1, 0.0, 0.6],
                   [-0.3, 0.1, 0.5, 0.2, -0.4]])          # (3,5)
A = torch.tensor([[0.3, -0.1, 0.2, 0.0, 0.5],
                  [-0.2, 0.4, 0.1, -0.3, 0.2]])            # (2,5) r=2
B = torch.tensor([[0.1, -0.2],
                  [0.3, 0.0],
                  [-0.1, 0.4]])                            # (3,2)
X = torch.tensor([[1.0, 0.5, -0.3, 0.2, 0.8],
                  [0.1, -0.4, 0.6, 0.3, -0.2]])            # (2,5)


def _oracle(x, W0, A, B, alpha, r):
    """独立重算：先合并 W' = W0 + (alpha/r) B A，再一次矩阵乘（结构异于参考解的分步旁路）。"""
    s = alpha / r
    Wp = W0 + s * (B @ A)
    return x @ Wp.T


def check_example(solve):
    y = solve(X, W0, A, B, 8.0, 2)
    exp = torch.tensor([[0.60999995470047, 1.6180000305175781, -0.8039999604225159],
                        [0.5160000324249268, -0.21199999749660492, -0.06600002199411392]])
    assert torch.allclose(y, exp, atol=1e-5, rtol=1e-5), f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(5)
    W = torch.randn(6, 4)
    a = torch.randn(3, 4)
    b = torch.randn(6, 3)
    x = torch.randn(5, 4)
    y = solve(x, W, a, b, alpha=16.0, r=3)
    exp = _oracle(x, W, a, b, 16.0, 3)
    assert torch.allclose(y, exp, atol=1e-5, rtol=1e-5), f"偏离参考:\n{y}\nvs\n{exp}"


def check_alpha_zero_is_base(solve):
    y = solve(X, W0, A, B, alpha=0.0, r=2)
    assert torch.allclose(y, X @ W0.T, atol=1e-6, rtol=1e-6), "alpha=0 应退化为纯基座输出"


def check_3d_batch(solve):
    x3 = torch.stack([X, X.flip(0)])          # (2,2,5)
    y = solve(x3, W0, A, B, 8.0, 2)
    assert y.shape == (2, 2, 3), f"3D 批量形状错: {tuple(y.shape)}"
    exp = _oracle(x3, W0, A, B, 8.0, 2)
    assert torch.allclose(y, exp, atol=1e-5, rtol=1e-5)


def check_matches_merged_equivalence(solve):
    y = solve(X, W0, A, B, 8.0, 2)
    s = 8.0 / 2
    y_merge = X @ (W0 + s * (B @ A)).T
    assert torch.allclose(y, y_merge, atol=1e-5, rtol=1e-5), "分步旁路应与合并权重等价"
