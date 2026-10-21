"""参考答案：Leaky ReLU —— where 掩码负区乘斜率。"""

import torch


def solve(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    return torch.where(x >= 0, x, negative_slope * x)
