"""参考答案：cosine / hinge 两种嵌入 margin 损失（逐元素与原生 F 对齐）。

口径：cosine 分母为两向量 2-范数乘积并 clamp_min(1e-8)，target=−1 支取
relu(cos − margin)；hinge 的 target=−1 支取 relu(margin − x1)；空集 mean
按本题兜底约定返回标量 0（原生会得到 NaN）。
"""

import torch


def solve(x1, x2, target, mode="cosine", margin=None, reduction="mean"):
    if mode == "cosine":
        if x2 is None:
            raise ValueError("mode='cosine' 需要 x2")
        m = 0.0 if margin is None else float(margin)
        dot = (x1 * x2).sum(dim=-1)
        denom = (x1.norm(p=2, dim=-1) * x2.norm(p=2, dim=-1)).clamp_min(1e-8)
        cos = dot / denom
        loss = torch.where(target == 1, 1.0 - cos, (cos - m).clamp_min(0.0))
    elif mode == "hinge":
        m = 1.0 if margin is None else float(margin)
        loss = torch.where(target == 1, x1, (m - x1).clamp_min(0.0))
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        if loss.numel() == 0:
            return torch.zeros((), dtype=loss.dtype)
        return loss.mean()
    raise ValueError(f"unknown reduction: {reduction!r}")
