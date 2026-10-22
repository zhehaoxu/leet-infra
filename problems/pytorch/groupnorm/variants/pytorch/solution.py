"""参考答案：GroupNorm —— 通道分组后 reshape，组内统计归一化，逐通道缩放平移。"""

import torch


def solve(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, num_groups: int, eps: float = 1e-5) -> torch.Tensor:
    n, c = x.shape[:2]
    spatial = x.shape[2:]
    # (N, C, *) -> (N, G, C//G, *)：每组内所有通道与空间位置一起统计
    xg = x.reshape(n, num_groups, c // num_groups, -1)
    mu = xg.mean(dim=(2, 3), keepdim=True)
    var = xg.var(dim=(2, 3), unbiased=False, keepdim=True)
    xn = (xg - mu) / torch.sqrt(var + eps)
    xn = xn.reshape(n, c, *spatial)
    shape = [1, c] + [1] * len(spatial)
    return xn * weight.reshape(shape) + bias.reshape(shape)
