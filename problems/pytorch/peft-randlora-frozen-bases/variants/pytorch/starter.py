import torch


def solve(W0, c, alpha, r, base_seed):
    """RandLora 合并：W' = W0 + (alpha/r) * B @ diag(c) @ A。
    B,A 由 base_seed 确定生成；B:(d_out,r), A:(r,d_in), c:(r,)。返回 W':(d_out,d_in)。"""
    # TODO
    raise NotImplementedError
