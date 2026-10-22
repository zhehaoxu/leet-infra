"""参考答案：WeightNorm 重参数化 —— w = g * v / ||v||_dim（范数下限 1e-12 防除零）。"""

import torch


def solve(weight_g: torch.Tensor, weight_v: torch.Tensor, dim: int = 0) -> torch.Tensor:
    norm = weight_v.norm(p=2, dim=dim, keepdim=True).clamp_min(1e-12)
    return weight_g * weight_v / norm
