"""参考答案：逐样本沿通道维求交叠/存在量，算 1 - soft Dice，按 reduction 归约。"""

import torch


def solve(prob, target, smoothing=1.0, reduction="mean"):
    inter = (prob * target).sum(dim=-1)
    denom = prob.sum(dim=-1) + target.sum(dim=-1)
    loss = 1.0 - (2 * inter + smoothing) / (denom + smoothing)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"unknown reduction: {reduction!r}")
