"""参考答案：Huber 损失 —— |z|<=delta 走二次、否则走线性，分段落 F.huber_loss。"""

import torch


def solve(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0, reduction: str = "mean") -> torch.Tensor:
    z = (pred - target).abs()
    quad = 0.5 * z * z
    lin = delta * (z - 0.5 * delta)
    loss = torch.where(z <= delta, quad, lin)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"unknown reduction: {reduction!r}")
