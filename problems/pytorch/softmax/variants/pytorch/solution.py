"""参考答案：数值稳定 Softmax —— 减 max 技巧。"""

import torch


def solve(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    m = logits.max(dim=dim, keepdim=True).values
    e = torch.exp(logits - m)  # exp 参数恒 <= 0，只下溢不会上溢
    return e / e.sum(dim=dim, keepdim=True)
