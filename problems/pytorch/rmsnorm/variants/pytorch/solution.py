"""参考答案：RMSNorm —— 不减均值，按尾部维的均方根缩放再乘 weight。"""

import torch


def solve(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    ms = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(ms + eps) * weight
