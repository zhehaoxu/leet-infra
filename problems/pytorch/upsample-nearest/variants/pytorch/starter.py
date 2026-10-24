import torch


def solve(x, scale):
    # TODO: 最近邻上采样。输出 (N, C, H*scale, W*scale)，
    # out[i, j] = x[i // scale, j // scale]（向下取整映射，等价 F.interpolate mode='nearest'）。
    raise NotImplementedError
