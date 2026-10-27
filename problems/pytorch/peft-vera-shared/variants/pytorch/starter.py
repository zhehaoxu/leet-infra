import torch


def solve(W0, lambda_b, lambda_d, A, B, x):
    """VeRA 前向：共享冻结基 A:(r,d_in)、B:(d_out,r)，每层可学缩放 lambda_d:(r,)、lambda_b:(d_out,)。
    y = x@W0^T + lambda_b ⊙ ((lambda_d ⊙ (x@A^T)) @ B^T)。
    返回 (y:(N,d_out) float32, n_trainable:int=d_out+r)。禁止现成 PEFT 库。"""
    # TODO
    raise NotImplementedError
