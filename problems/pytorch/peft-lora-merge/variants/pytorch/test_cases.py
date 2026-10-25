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


def _oracle(W0, A, B, alpha, r):
    """独立重算：三重循环逐元素 W0[i,j] + (alpha/r)·Σ_k B[i,k]A[k,j]（结构异于矩阵乘）。"""
    s = alpha / r
    d_out, d_in = W0.shape
    out = W0.clone()
    for i in range(d_out):
        for j in range(d_in):
            acc = 0.0
            for k in range(A.shape[0]):
                acc += B[i, k].item() * A[k, j].item()
            out[i, j] = W0[i, j] + s * acc
    return out


def check_example(solve):
    Wp = solve(W0, A, B, 8.0, 2)
    exp = torch.tensor([[0.7799999713897705, -0.5600000023841858, 0.10000000149011612, 0.5400000214576721, 0.03999999165534973],
                        [0.5600000023841858, 0.2800000011920929, 0.14000001549720764, 0.0, 1.2000000476837158],
                        [-0.7400000095367432, 0.7800000309944153, 0.5799999833106995, -0.2800000309944153, -0.2799999713897705]])
    assert torch.allclose(Wp, exp, atol=1e-5, rtol=1e-5), f"got {Wp.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(6)
    W = torch.randn(4, 5)
    a = torch.randn(2, 5)
    b = torch.randn(4, 2)
    Wp = solve(W, a, b, alpha=10.0, r=2)
    assert torch.allclose(Wp, _oracle(W, a, b, 10.0, 2), atol=1e-5, rtol=1e-5)


def check_alpha_zero_identity(solve):
    Wp = solve(W0, A, B, alpha=0.0, r=2)
    assert torch.equal(Wp, W0), "alpha=0 时合并权重应恒等 W0"


def check_apply_equivalence(solve):
    Wp = solve(W0, A, B, 8.0, 2)
    s = 8.0 / 2
    y_split = X @ W0.T + s * ((X @ A.T) @ B.T)     # LoRA 分步前向
    assert torch.allclose(X @ Wp.T, y_split, atol=1e-5, rtol=1e-5), "合并权重应与分步前向逐元素等价"


def check_shape(solve):
    Wp = solve(W0, A, B, 8.0, 2)
    assert Wp.shape == W0.shape, f"合并权重形状应与 W0 同: {tuple(Wp.shape)}"
