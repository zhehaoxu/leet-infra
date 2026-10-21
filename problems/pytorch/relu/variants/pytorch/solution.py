"""参考答案：ReLU —— 向量化实现，不碰 Python 循环。"""

import torch


def solve(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, torch.zeros_like(x))
