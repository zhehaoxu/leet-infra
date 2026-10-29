import torch

W0 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])          # (3,2)
THETA = torch.arange(1, 11, dtype=torch.float32)                  # (10,)


def _oracle(W0, theta, r, alpha, proj_seed, layer_id=0):
    """独立重算：用标量三重循环按同样索引分配重建 ΔW。"""
    d_out, d_in = W0.shape
    n_a = r * d_in
    n_b = r * d_out
    n_params = n_a + n_b
    D = theta.numel()
    g = torch.Generator()
    g.manual_seed(int(proj_seed) + int(layer_id))
    perm = torch.randperm(D, generator=g)
    selected = perm[:n_params]
    A_vals = theta[selected[:n_a]].tolist()
    B_vals = theta[selected[n_a:]].tolist()
    scale = float(alpha) / float(r)
    merged = W0.clone().to(torch.float64)
    for i in range(d_out):
        for j in range(d_in):
            Bij = [B_vals[i * r + u] for u in range(r)]
            Aij = [A_vals[u * d_in + j] for u in range(r)]
            delta = sum(Bij[u] * Aij[u] for u in range(r)) * scale
            merged[i, j] += delta
    return merged.to(torch.float32)


def check_example(solve):
    wp = solve(W0, THETA, 1, 2.0, 7, 0)
    exp = torch.tensor([[49.0, 10.0], [63.0, 14.0], [29.0, 10.0]])
    assert torch.allclose(wp, exp, atol=1e-5, rtol=1e-5), f"got {wp.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(13)
    w0 = torch.randn(4, 5)
    theta = torch.randn(30)
    wp = solve(w0, theta, 2, 3.0, 123, 1)
    assert torch.allclose(wp, _oracle(w0, theta, 2, 3.0, 123, 1),
                          atol=1e-5, rtol=1e-5), "与独立索引重算不符"


def check_alpha_zero_noop(solve):
    wp = solve(W0, THETA, 1, 0.0, 7, 0)
    assert torch.allclose(wp, W0, atol=1e-6, rtol=1e-6), "alpha=0 应退化为 W0"


def check_theta_zeros_noop(solve):
    wp = solve(W0, torch.zeros(10), 1, 2.0, 7, 0)
    assert torch.allclose(wp, W0, atol=1e-6, rtol=1e-6), "theta 选中项全零时应退化为 W0"


def check_layer_id_changes_assignment(solve):
    """同一 proj_seed 下不同 layer_id 应得到不同 A/B 切片。"""
    wp0 = solve(W0, THETA, 1, 2.0, 7, 0)
    wp1 = solve(W0, THETA, 1, 2.0, 7, 1)
    assert not torch.allclose(wp0, wp1, atol=1e-4, rtol=1e-4), "layer_id=0/1 应产生不同 W'"
