import torch


def solve(W0, freqs, coeffs, alpha):
    """FourierFT 合并：W' = W0 + alpha * sum_{(u,v)} c * cos(2*pi*(u*i/M+v*j/N))。
    W0:(d_out,d_in) freqs:(m,2) coeffs:(m,)。返回 W':(d_out,d_in)。"""
    d_out, d_in = W0.shape
    delta = torch.zeros_like(W0)
    rows = torch.arange(d_out, dtype=torch.float32, device=W0.device)[:, None]
    cols = torch.arange(d_in, dtype=torch.float32, device=W0.device)[None, :]
    for (u, v), c in zip(freqs.tolist(), coeffs.tolist()):
        phase = 2 * torch.pi * (rows * u / d_out + cols * v / d_in)
        delta += c * torch.cos(phase)
    return W0 + alpha * delta
