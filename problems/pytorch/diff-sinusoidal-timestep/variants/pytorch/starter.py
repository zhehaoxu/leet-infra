import torch


def solve(t, dim):
    """正弦时间步嵌入：half=dim//2; freq_i=10000**(-i/half); angles=t*freq;
    返回 cat([sin(angles), cos(angles)])，形状 (dim,)。t 为标量，dim 为偶数。禁止现成扩散库。"""
    # TODO
    raise NotImplementedError
