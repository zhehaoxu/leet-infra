import torch


def solve(W, X):
    """BitNet b1.58：absmean 缩放 + 三值量化 + 加减计数 GEMM。"""
    s = float(W.abs().mean())
    if s == 0:
        T = torch.zeros_like(W, dtype=torch.int64)
        W_hat = torch.zeros_like(W)
        Y = torch.zeros(X.shape[0], W.shape[0], dtype=W.dtype, device=W.device)
        return s, T, W_hat, Y
    T = torch.clamp(torch.round(W / s), -1, 1).to(torch.int64)
    W_hat = T.to(W.dtype) * s
    pos = (T == 1).to(W.dtype)
    neg = (T == -1).to(W.dtype)
    Y = s * (X @ pos.T - X @ neg.T)
    return s, T, W_hat, Y
