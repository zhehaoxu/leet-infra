import torch


def solve(packed, scales, zeros, group_size, N):
    """per-group 非对称 uint4 解包：低 4 位偶数索引，高 4 位奇数索引。

    返回浮点权重 W。
    """
    # TODO: 解包 uint8 缓冲并按 group_size 反量化
    raise NotImplementedError
