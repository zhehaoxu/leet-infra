"""参考答案：InstanceNorm —— 逐样本逐通道，对空间维（dim>=2）归约统计。"""

import torch


def solve(x: torch.Tensor, weight: torch.Tensor | None = None, bias: torch.Tensor | None = None, eps: float = 1e-5) -> torch.Tensor:
    dims = tuple(range(2, x.dim()))  # 空间维；不含 batch，也不跨通道
    mu = x.mean(dims, keepdim=True)
    var = x.var(dims, unbiased=False, keepdim=True)
    y = (x - mu) / torch.sqrt(var + eps)
    if weight is not None or bias is not None:
        shape = [1, -1] + [1] * (x.dim() - 2)
        if weight is not None:
            y = y * weight.reshape(shape)
        if bias is not None:
            y = y + bias.reshape(shape)
    return y
