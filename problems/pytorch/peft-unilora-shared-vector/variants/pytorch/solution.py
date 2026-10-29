import torch


def solve(W0, theta, r, alpha, proj_seed, layer_id=0):
    """UniLoRA 合并：从共享向量 theta 中按 proj_seed+layer_id 固定索引取出 A/B，
    计算 W' = W0 + (alpha/r) * B @ A。theta: (D,) W0:(d_out,d_in)。返回 W':(d_out,d_in)。"""
    d_out, d_in = W0.shape
    n_a = r * d_in
    n_b = r * d_out
    n_params = n_a + n_b
    D = theta.numel()
    if n_params > D:
        raise ValueError("theta 长度不足以分配本层 A/B")
    g = torch.Generator(device=theta.device)
    g.manual_seed(int(proj_seed) + int(layer_id))
    perm = torch.randperm(D, generator=g, device=theta.device)
    selected = perm[:n_params]
    A = theta[selected[:n_a]].reshape(r, d_in)
    B = theta[selected[n_a:]].reshape(d_out, r)
    scale = float(alpha) / float(r)
    return W0 + scale * (B @ A)
