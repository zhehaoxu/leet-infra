"""参考答案：GLU —— 沿 dim 对半切成 a、b，返回 a * sigmoid(b)。"""

import torch


def solve(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    a, b = torch.chunk(x, 2, dim=dim)
    return a * torch.sigmoid(b)
