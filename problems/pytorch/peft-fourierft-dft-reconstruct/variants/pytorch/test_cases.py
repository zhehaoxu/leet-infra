import torch

W0 = torch.zeros(2, 2, dtype=torch.float32)
FREQS = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.int64)
COEFFS = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)


def _oracle(W0, freqs, coeffs, alpha):
    """独立重算：逐位置按公式求和。"""
    d_out, d_in = W0.shape
    merged = W0.clone().to(torch.float64)
    for i in range(d_out):
        for j in range(d_in):
            val = 0.0
            for (u, v), c in zip(freqs.tolist(), coeffs.tolist()):
                val += c * torch.cos(torch.tensor(2 * torch.pi * (u * i / d_out + v * j / d_in))).item()
            merged[i, j] += alpha * val
    return merged.to(torch.float32)


def check_example(solve):
    wp = solve(W0, FREQS, COEFFS, 1.0)
    exp = torch.tensor([[1.75, 1.25], [0.75, 0.25]])
    assert torch.allclose(wp, exp, atol=1e-5, rtol=1e-5), f"got {wp.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(29)
    w0 = torch.randn(3, 4)
    freqs = torch.tensor([[0, 0], [1, 2], [2, 1], [2, 3]], dtype=torch.int64)
    coeffs = torch.randn(4)
    wp = solve(w0, freqs, coeffs, 0.5)
    assert torch.allclose(wp, _oracle(w0, freqs, coeffs, 0.5),
                          atol=1e-5, rtol=1e-5), "与独立逐位置重算不符"


def check_alpha_zero_noop(solve):
    wp = solve(W0, FREQS, COEFFS, 0.0)
    assert torch.allclose(wp, W0, atol=1e-6, rtol=1e-6), "alpha=0 时应退化为 W0"


def check_empty_freqs_noop(solve):
    wp = solve(W0, torch.empty(0, 2, dtype=torch.int64), torch.empty(0), 1.0)
    assert torch.allclose(wp, W0, atol=1e-6, rtol=1e-6), "空频率列表时应退化为 W0"


def check_dc_only_uniform_shift(solve):
    """只有 (0,0) 频率时，delta 是常数矩阵。"""
    wp = solve(W0, torch.tensor([[0, 0]], dtype=torch.int64), torch.tensor([2.0]), 1.0)
    exp = torch.full((2, 2), 2.0)
    assert torch.allclose(wp, exp, atol=1e-6, rtol=1e-6), "仅 DC 分量应产生均匀平移"
