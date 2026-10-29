import torch


def solve(W0, r, alpha, bits, iterations):
    """LoftQ 合并：交替量化 + SVD 初始化后返回 W_q + (alpha/r)*B@A。
    W0:(d_out,d_in) bits>0 iterations>=0。返回 W':(d_out,d_in)。"""
    # TODO
    raise NotImplementedError
