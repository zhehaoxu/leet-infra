import torch


def solve(W0, A, B, alpha, r):
    """LoRA 权重合并：W' = W0 + (alpha/r) * (B @ A)。
    W0:(d_out,d_in) A:(r,d_in) B:(d_out,r)。返回与 W0 同形的合并权重。禁止现成 PEFT 库。"""
    # TODO
    raise NotImplementedError
