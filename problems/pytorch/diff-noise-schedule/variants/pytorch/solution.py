"""参考答案：线性 / 余弦噪声调度，输出 alpha_bar 序列。"""

import math

import torch


def solve(T, kind):
    if kind == "linear":
        betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float32)
    elif kind == "cosine":
        s = 0.008
        x = torch.linspace(0, T, T + 1, dtype=torch.float32)
        f = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        ac = f / f[0]
        betas = 1 - ac[1:] / ac[:-1]
        betas = torch.clamp(betas, 0.0, 0.999)
    else:
        raise ValueError(f"unknown kind: {kind}")
    return torch.cumprod(1 - betas, dim=0)
