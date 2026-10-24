"""参考答案：二元交叉熵损失 —— 输入为概率 (0,1)，稳定 clamp 后按公式逐元素。"""

import torch


def solve(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    # 把概率夹到开区间，避免 log(0)=-inf；内部值不受影响
    eps = torch.tensor(1e-12, dtype=pred.dtype, device=pred.device)
    p = torch.clamp(pred, min=float(eps), max=1.0 - float(eps))
    loss = -(target * torch.log(p) + (1.0 - target) * torch.log(1.0 - p))
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"unknown reduction: {reduction!r}")
