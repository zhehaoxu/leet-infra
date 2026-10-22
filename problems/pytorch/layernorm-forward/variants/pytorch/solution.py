"""参考答案：LayerNorm 前向 —— 沿尾部若干维做无偏=False 的统计归一化。"""

import torch


def solve(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    n_norm = weight.dim()  # 归一化尾部 n_norm 个维度
    norm_dims = tuple(range(x.dim() - n_norm, x.dim()))
    mu = x.mean(norm_dims, keepdim=True)
    var = x.var(norm_dims, unbiased=False, keepdim=True)  # LayerNorm 用有偏方差
    return (x - mu) / torch.sqrt(var + eps) * weight + bias
