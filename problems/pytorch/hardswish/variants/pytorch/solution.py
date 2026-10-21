"""参考答案：HardSwish —— x * relu6(x+3) / 6。"""

import torch


def solve(x: torch.Tensor) -> torch.Tensor:
    # F.hardswish = x * clamp(x+3, 0, 6) / 6
    return x * torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0
