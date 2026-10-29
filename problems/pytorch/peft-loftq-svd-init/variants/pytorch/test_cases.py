import torch

W0 = torch.tensor([[0.1, 0.6, -0.3], [0.8, -0.2, 0.4]], dtype=torch.float32)


def _oracle(W0, r, alpha, bits, iterations):
    """独立重算：使用 torch.svd 走不同 SVD 后端，最终合并权重应一致。"""

    def quantize(W, bits):
        scale = W.abs().max()
        if scale == 0:
            return W.clone()
        qmax = 2 ** (bits - 1) - 1
        return torch.round(W / scale * qmax) / qmax * scale

    Wq = quantize(W0, bits)
    B = torch.zeros(W0.size(0), r, device=W0.device, dtype=W0.dtype)
    A = torch.zeros(r, W0.size(1), device=W0.device, dtype=W0.dtype)
    for _ in range(iterations):
        R = W0 - Wq
        U, S, V = torch.svd(R)
        S_r = S[:r].sqrt()
        B = U[:, :r] * S_r
        A = S_r[:, None] * V[:, :r].t()
        Wq = quantize(W0 - B @ A, bits)
    scale = float(alpha) / float(r) if r > 0 else 0.0
    return Wq + scale * (B @ A)


def check_example(solve):
    wp = solve(W0, 1, 1.0, 2, 1)
    exp = torch.tensor([[0.03244384378194809, 0.8755617737770081, -0.28459733724594116],
                        [0.800000011920929, -0.0414794459939003, 0.41067415475845337]])
    assert torch.allclose(wp, exp, atol=1e-5, rtol=1e-5), f"got {wp.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(23)
    w0 = torch.randn(4, 5)
    wp = solve(w0, 2, 1.5, 4, 2)
    assert torch.allclose(wp, _oracle(w0, 2, 1.5, 4, 2),
                          atol=1e-4, rtol=1e-4), "与独立 SVD 重算不符"


def check_iterations_zero_pure_quantize(solve):
    wp = solve(W0, 1, 1.0, 2, 0)
    # absmax scale = 0.8, qmax = 1, so each entry rounds to {-1, 0, 1} * 0.8
    exp = torch.tensor([[0.0, 0.8, 0.0], [0.8, 0.0, 0.0]])
    assert torch.allclose(wp, exp, atol=1e-6, rtol=1e-6), "iterations=0 应等于对称 absmax 量化结果"


def check_alpha_zero_returns_quantized_base(solve):
    wp = solve(W0, 1, 0.0, 2, 1)
    # alpha=0 应只返回最终量化基座
    exp = _oracle(W0, 1, 0.0, 2, 1)
    assert torch.allclose(wp, exp, atol=1e-5, rtol=1e-5), "alpha=0 应只返回量化基座"


def check_zero_matrix_stays_zero(solve):
    wp = solve(torch.zeros(2, 3), 1, 1.0, 2, 1)
    assert torch.allclose(wp, torch.zeros(2, 3), atol=1e-6, rtol=1e-6), "全零 W0 应保持全零"
