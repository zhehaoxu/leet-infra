"""参考答案：手写 LayerNorm 前向。"""

import torch


def solve(x: torch.Tensor, weight=None, bias=None, eps: float = 1e-5) -> torch.Tensor:
    mu = x.mean(dim=-1, keepdim=True)
    var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)  # 有偏方差
    x_hat = (x - mu) / torch.sqrt(var + eps)
    if weight is not None:
        x_hat = x_hat * weight
    if bias is not None:
        x_hat = x_hat + bias
    return x_hat
