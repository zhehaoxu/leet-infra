"""参考答案：Tanh —— 按符号分支用 exp(-2|x|) 避免上溢。"""

import torch


def solve(x: torch.Tensor) -> torch.Tensor:
    # exp(2x) 在大正数会溢出为 inf 导致 inf/inf=NaN；改用 exp(-2|x|)（恒 <=1）
    pos = x >= 0
    ex = torch.exp(-2 * x.abs())
    t = (1 - ex) / (1 + ex)
    return torch.where(pos, t, -t)
