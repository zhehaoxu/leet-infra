import torch


def solve(x, axis):
    # TODO: 沿 axis 的累积和（不用 torch.cumsum）。
    # out[0]=x[0]; out[i]=out[i-1]+x[i]（沿该轴递推）。负轴需归一化。
    raise NotImplementedError
