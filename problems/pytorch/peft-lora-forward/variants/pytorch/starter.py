import torch


def solve(x, W0, A, B, alpha, r):
    """LoRA 前向：y = x @ W0.T + (alpha/r) * (x @ A.T) @ B.T。
    x:(...,d_in) W0:(d_out,d_in) A:(r,d_in) B:(d_out,r)。返回 (...,d_out)。禁止现成 PEFT 库。"""
    # TODO
    raise NotImplementedError
