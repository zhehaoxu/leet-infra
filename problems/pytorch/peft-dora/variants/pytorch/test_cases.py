import torch

W0 = torch.tensor([[0.5, -0.2, 0.1, 0.3, 0.0],
                   [0.2, 0.4, -0.1, 0.0, 0.6],
                   [-0.3, 0.1, 0.5, 0.2, -0.4]])      # (3,5)
A = torch.tensor([[0.3, -0.1, 0.2, 0.0, 0.5],
                  [-0.2, 0.4, 0.1, -0.3, 0.2]])       # (2,5) r=2
B = torch.tensor([[0.1, -0.2],
                  [0.3, 0.0],
                  [-0.1, 0.4]])                       # (3,2)
X = torch.tensor([[1.0, 0.5, -0.3, 0.2, 0.8],
                  [0.1, -0.4, 0.6, 0.3, -0.2]])       # (2,5)
M = torch.tensor([0.7, 0.8, 0.6])                     # (3,) d_out
ALPHA, R = 8.0, 2


def _oracle(x, W0, A, B, m, alpha, r):
    """独立重算：显式逐输出行 loop 计算范数与幅值-方向合成（结构异于参考解的批量归一化）。"""
    s = alpha / r
    V = W0 + s * (B @ A)
    d_out, d_in = V.shape
    Weff = torch.zeros_like(V)
    for i in range(d_out):
        row = V[i]
        n_i = torch.sqrt((row * row).sum())
        Weff[i] = m[i] * row / n_i
    return x @ Weff.T


def check_example(solve):
    y = solve(X, W0, A, B, M, ALPHA, R)
    exp = torch.tensor([[0.38576632738113403, 0.9512505531311035, -0.3756389617919922],
                        [0.32632043957710266, -0.12463850528001785, -0.0308360755443573]])
    assert torch.allclose(y, exp, atol=1e-5, rtol=1e-5), f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(11)
    x = torch.randn(6, 4)
    w0 = torch.randn(3, 4)
    a = torch.randn(2, 4)
    b = torch.randn(3, 2)
    m = torch.rand(3) + 0.5
    y = solve(x, w0, a, b, m, 6.0, 2)
    assert torch.allclose(y, _oracle(x, w0, a, b, m, 6.0, 2), atol=1e-5, rtol=1e-5)


def check_m_equals_vnorm_reduces_to_lora(solve):
    # m = ||V||_row 时 Weff==V，DoRA 前向 == 合并 LoRA 后前向
    s = ALPHA / R
    V = W0 + s * (B @ A)
    m = V.norm(p=2, dim=1)
    y = solve(X, W0, A, B, m, ALPHA, R)
    y_lora = X @ V.T
    assert torch.allclose(y, y_lora, atol=1e-5, rtol=1e-5), "m=||V||row 应退化为合并 LoRA 前向"


def check_alpha_zero_recovers_base(solve):
    # alpha=0 且 m=||W0||_row 时回到基座前向 x @ W0.T
    m = W0.norm(p=2, dim=1)
    y = solve(X, W0, A, B, m, 0.0, R)
    assert torch.allclose(y, X @ W0.T, atol=1e-5, rtol=1e-5), "alpha=0,m=||W0|| 应回到基座前向"


def check_3d_batch(solve):
    x3 = torch.stack([X, X.flip(0)])          # (2,2,5)
    y = solve(x3, W0, A, B, M, ALPHA, R)
    assert y.shape == (2, 2, 3), f"3D 批量形状错: {tuple(y.shape)}"
    assert torch.allclose(y, _oracle(x3, W0, A, B, M, ALPHA, R), atol=1e-5, rtol=1e-5)
