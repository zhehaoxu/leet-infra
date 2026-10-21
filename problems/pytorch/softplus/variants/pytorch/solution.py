"""参考答案：SoftPlus —— 稳定 log(1+exp(beta*x))/beta，大数走线性分支。"""

import torch


def solve(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor:
    # beta*x 超过 threshold 时 softplus 逼近 x（与 PyTorch 定义一致），避免 exp 上溢
    bx = beta * x
    return torch.where(bx > threshold, x, torch.log1p(torch.exp(bx)) / beta)
