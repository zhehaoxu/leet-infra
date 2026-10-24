import torch


def solve(x, axis, keepdim=False):
    # TODO: 沿 axis 求和（不用 torch.sum）。负轴需归一化。
    # 把 axis 上的每一片 acc += x[..., i, ...] 累加；keepdim 时 unsqueeze 回该轴。
    raise NotImplementedError
