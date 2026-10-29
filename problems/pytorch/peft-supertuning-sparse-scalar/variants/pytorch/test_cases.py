import torch

W0 = torch.tensor([[1.0, -3.0, 2.0], [0.5, -3.0, 1.0]])    # (2,3)
S = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
A = torch.tensor([[1.0, 0.0, -1.0]])                        # (1,3)
B = torch.tensor([[2.0], [1.0]])                            # (2,1)


def _oracle(W0, S, k, A, B, alpha, r):
    """独立重算：用标量循环构建 top-k 掩码并求和。"""
    d_out, d_in = W0.shape
    flat_abs = W0.abs().view(-1).tolist()
    order = sorted(range(len(flat_abs)), key=lambda i: (-flat_abs[i], i))
    mask_set = set(order[:k])
    merged = W0.clone().to(torch.float64)
    for i in range(d_out):
        for j in range(d_in):
            flat_idx = i * d_in + j
            if flat_idx in mask_set:
                merged[i, j] += float(S[i, j])
            if r > 0:
                scale = float(alpha) / float(r)
                lora = sum(float(B[i, u]) * float(A[u, j]) for u in range(r)) * scale
                merged[i, j] += lora
    return merged.to(torch.float32)


def check_example(solve):
    wp = solve(W0, S, 2, A, B, 2.0, 1)
    exp = torch.tensor([[5.0, -2.8, -2.0], [2.5, -2.5, -1.0]])
    assert torch.allclose(wp, exp, atol=1e-5, rtol=1e-5), f"got {wp.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(17)
    w0 = torch.randn(3, 4)
    s = torch.randn(3, 4)
    a = torch.randn(2, 4)
    b = torch.randn(3, 2)
    wp = solve(w0, s, 5, a, b, 3.0, 2)
    assert torch.allclose(wp, _oracle(w0, s, 5, a, b, 3.0, 2),
                          atol=1e-5, rtol=1e-5), "与独立逐元素重算不符"


def check_k_zero_lora_only(solve):
    wp = solve(W0, S, 0, A, B, 2.0, 1)
    exp = W0 + 2.0 * (B @ A)
    assert torch.allclose(wp, exp, atol=1e-6, rtol=1e-6), "k=0 时应只保留 LoRA 项"


def check_r_zero_scalar_only(solve):
    wp = solve(W0, S, 2, A, B, 2.0, 0)
    exp = torch.tensor([[1.0, -2.8, 2.0], [0.5, -2.5, 1.0]])
    assert torch.allclose(wp, exp, atol=1e-6, rtol=1e-6), "r=0 时应只保留稀疏标量项"


def check_alpha_zero_no_lora(solve):
    wp = solve(W0, S, 2, A, B, 0.0, 1)
    exp = torch.tensor([[1.0, -2.8, 2.0], [0.5, -2.5, 1.0]])
    assert torch.allclose(wp, exp, atol=1e-6, rtol=1e-6), "alpha=0 时 LoRA 项应为零"


def check_tie_break_smallest_index(solve):
    """|W0| 在 (0,1) 与 (1,1) 处同为 3.0，稳定排序应优先取 (0,1)。"""
    wp = solve(W0, S, 1, torch.zeros(0, 3), torch.zeros(2, 0), 0.0, 0)
    exp = W0.clone()
    exp[0, 1] += S[0, 1]
    assert torch.allclose(wp, exp, atol=1e-6, rtol=1e-6), "并列时应按下标小者优先"
