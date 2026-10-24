import torch


def solve(W, x_bar, alphas, bits):
    """AWQ：s_j=x_bar_j^alpha 归一化→放大 per-tensor RTN→缩回；网格选最小加权输出误差。"""
    # TODO
    raise NotImplementedError
