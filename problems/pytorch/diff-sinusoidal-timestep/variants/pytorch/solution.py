"""参考答案：正弦时间步嵌入（前半 sin、后半 cos）。"""

import math

import torch


def solve(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    angles = t * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)])
