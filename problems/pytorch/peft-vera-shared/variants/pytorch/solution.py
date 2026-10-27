import torch


def solve(W0, lambda_b, lambda_d, A, B, x):
    """VeRA 前向：共享冻结基 A:(r,d_in)、B:(d_out,r)，每层可学缩放 lambda_d:(r,)、lambda_b:(d_out,)。
    y = x@W0^T + lambda_b ⊙ ((lambda_d ⊙ (x@A^T)) @ B^T)。
    返回 (y:(N,d_out) float32, n_trainable:int=d_out+r)。禁止现成 PEFT 库。"""
    x64 = x.to(torch.float64)
    base = x64 @ W0.to(torch.float64).T
    h = (x64 @ A.to(torch.float64).T) * lambda_d.to(torch.float64)
    u = (h @ B.to(torch.float64).T) * lambda_b.to(torch.float64)
    y = base + u
    n_trainable = int(W0.shape[0] + lambda_d.numel())
    return y.to(torch.float32), n_trainable
