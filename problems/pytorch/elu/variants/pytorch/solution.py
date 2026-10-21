"""参考答案：ELU —— 负区用 expm1 保近零精度。"""

import torch


def solve(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    # expm1(x)=exp(x)-1 在 x 接近 0 时比 exp(x)-1 精度更高
    return torch.where(x >= 0, x, alpha * torch.expm1(x))
