"""参考答案：HardSigmoid —— relu6(x+3) / 6，截断到 [0,1]。"""

import torch


def solve(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0
