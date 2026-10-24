"""参考答案：MSE 损失 —— (pred-target)^2 逐元素，按 reduction 归约。"""

import torch


def solve(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    diff = pred - target
    loss = diff * diff
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"unknown reduction: {reduction!r}")
