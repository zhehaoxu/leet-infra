"""参考答案：F.binary_cross_entropy_with_logits 等价实现。

推导：BCE = w·y·log(1+e^{−x}) + (1−y)·log(1+e^{x})，而
log(1+e^{−x}) = (−x)⁺ + L、log(1+e^{x}) = x⁺ + L、L = log(1+e^{−|x|}) = log1p(e^{−|x|})
在 |x| 很大时 e^{−|x|} 下溢为 0、L 精确取 0，正是它的双侧渐近值，因此全程有限。

mean 的分母固定为 numel，不是 Σweight——与 F.binary_cross_entropy 的 "mean" 口径一致。
"""

import torch


def solve(logits, targets, weight=None, reduction="mean", pos_weight=None):
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"未知 reduction: {reduction!r}")
    x = logits.double()
    y = targets.double()
    w = 1.0 if pos_weight is None else pos_weight.double()

    lsoftplus = torch.log1p((-x.abs()).exp())      # L = log(1 + e^{-|x|})
    xpos = x.clamp_min(0.0)
    loss = (1.0 - y) * xpos + w * y * (xpos - x) + (1.0 + y * (w - 1.0)) * lsoftplus

    if weight is not None:
        loss = loss * weight.double()

    if reduction == "none":
        return loss.to(logits.dtype)
    total = loss.sum()
    if reduction == "sum":
        return total.to(logits.dtype)
    return (total / logits.numel()).to(logits.dtype)
