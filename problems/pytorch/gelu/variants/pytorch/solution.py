"""参考答案：GELU —— erf 精确式与 tanh 近似式。"""

import math

import torch


def solve(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    if approximate == "none":
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
    if approximate == "tanh":
        inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        return 0.5 * x * (1.0 + torch.tanh(inner))
    raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")
