"""参考答案：Mish —— x * tanh(softplus(x))，softplus 稳定计算。"""

import torch


def solve(x: torch.Tensor) -> torch.Tensor:
    # softplus(x) = log(1+exp(x))，按符号分支避免上溢
    sp = torch.where(x > 0, x + torch.log1p(torch.exp(-x)), torch.log1p(torch.exp(x)))
    return x * torch.tanh(sp)
