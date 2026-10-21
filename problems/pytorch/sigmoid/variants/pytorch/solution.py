"""参考答案：Sigmoid —— 按符号分支的数值稳定实现。"""

import torch


def solve(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    pos = x >= 0
    neg = ~pos
    # x >= 0: 1 / (1 + exp(-x))，exp 参数非正，不会上溢
    out[pos] = 1.0 / (1.0 + torch.exp(-x[pos]))
    # x < 0: exp(x) / (1 + exp(x))，同样让 exp 参数非正
    ex = torch.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out
