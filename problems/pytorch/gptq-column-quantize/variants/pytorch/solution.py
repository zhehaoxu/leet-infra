import torch


def solve(W, H, bits, damp):
    """GPTQ：per-tensor 对称量化 + 逐列 Hessian 逆误差补偿，重构回 (O, C)。"""
    O, C = W.shape
    qmax = 2 ** (bits - 1) - 1
    scale = float(W.abs().max()) or 1.0
    Hinv = torch.linalg.inv(H + damp * torch.eye(C, dtype=H.dtype))
    Wq = W.clone()
    out = W.clone()
    for i in range(C):
        col = Wq[:, i]
        q = torch.clamp(torch.round(col / scale * qmax), -qmax - 1, qmax)
        dq = q / qmax * scale
        err = col - dq
        out[:, i] = dq
        d = Hinv[i, i]
        Wq[:, i + 1:] -= (err / d).unsqueeze(1) * Hinv[i, i + 1:].unsqueeze(0)
    return out
