"""参考答案：Swish / SiLU —— x * sigmoid(x)。"""

import torch


def solve(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)
