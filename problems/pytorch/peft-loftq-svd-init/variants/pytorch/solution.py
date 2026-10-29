import torch


def solve(W0, r, alpha, bits, iterations):
    """LoftQ 合并：交替量化 + SVD 初始化后返回 W_q + (alpha/r)*B@A。
    W0:(d_out,d_in) bits>0 iterations>=0。返回 W':(d_out,d_in)。"""

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
        U, S, Vh = torch.linalg.svd(R, full_matrices=False)
        S_r = S[:r].sqrt()
        B = U[:, :r] * S_r
        A = S_r[:, None] * Vh[:r, :]
        Wq = quantize(W0 - B @ A, bits)
    scale = float(alpha) / float(r) if r > 0 else 0.0
    return Wq + scale * (B @ A)
