import torch

W0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])     # (2,2)
C = torch.tensor([2.0])                          # r=1


def _oracle(W0, c, alpha, r, base_seed):
    """独立重算：逐标量按同样随机序列生成 B/A 后三重循环求 delta。"""
    d_out, d_in = W0.shape
    g = torch.Generator()
    g.manual_seed(int(base_seed))
    B = torch.randn(d_out, r, generator=g)
    A = torch.randn(r, d_in, generator=g)
    scale = float(alpha) / float(r)
    merged = W0.clone().to(torch.float64)
    for i in range(d_out):
        for j in range(d_in):
            val = 0.0
            for u in range(r):
                val += float(B[i, u]) * float(c[u]) * float(A[u, j])
            merged[i, j] += scale * val
    return merged.to(torch.float32)


def check_example(solve):
    wp = solve(W0, C, 1.0, 1, 42)
    exp = torch.tensor([[1.1578824520111084, 2.155101776123047],
                        [3.0604019165039062, 4.059338092803955]])
    assert torch.allclose(wp, exp, atol=1e-5, rtol=1e-5), f"got {wp.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(19)
    w0 = torch.randn(3, 4)
    c = torch.randn(2)
    wp = solve(w0, c, 2.5, 2, 12345)
    assert torch.allclose(wp, _oracle(w0, c, 2.5, 2, 12345),
                          atol=1e-5, rtol=1e-5), "与独立逐元素重算不符"


def check_zero_coefficients_noop(solve):
    wp = solve(W0, torch.zeros(1), 1.0, 1, 42)
    assert torch.allclose(wp, W0, atol=1e-6, rtol=1e-6), "c=0 时应退化为 W0"


def check_alpha_zero_noop(solve):
    wp = solve(W0, C, 0.0, 1, 42)
    assert torch.allclose(wp, W0, atol=1e-6, rtol=1e-6), "alpha=0 时应退化为 W0"


def check_different_seed_different_bases(solve):
    """不同 base_seed 应产生不同随机基，从而 W' 不同。"""
    wp1 = solve(W0, C, 1.0, 1, 42)
    wp2 = solve(W0, C, 1.0, 1, 43)
    assert not torch.allclose(wp1, wp2, atol=1e-3, rtol=1e-3), "不同 seed 应产生不同 W'"
