"""参考答案：GIoU / DIoU / CIoU 三种 IoU 变体损失（逐对向量化，纯 torch 重算）。

口径：基础 IoU 同 iou-compute（交集宽高 clamp_min(0)，并集为 0 记 IoU=0）；
GIoU/DIoU/CIoU 的各惩罚项在分母为 0 的退化情形下记 0，保证全路径有限。
"""

import math

import torch


def _safe_div(a, b, mask):
    """mask 为 True 处做 a/b，False 处记 0（b 在 False 处可能是 0，先换成分母 1 防 NaN）。"""
    return torch.where(mask, a / torch.where(mask, b, torch.ones_like(b)), torch.zeros_like(a))


def solve(pred, target, reduction="mean"):
    ax1, ay1, ax2, ay2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    bx1, by1, bx2, by2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    # 基础 IoU（同 iou-compute 口径）
    iw = (torch.minimum(ax2, bx2) - torch.maximum(ax1, bx1)).clamp_min(0.0)
    ih = (torch.minimum(ay2, by2) - torch.maximum(ay1, by1)).clamp_min(0.0)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    iou = _safe_div(inter, union, union > 0)

    # 最小包围框 C（enclose）
    ew = (torch.maximum(ax2, bx2) - torch.minimum(ax1, bx1)).clamp_min(0.0)
    eh = (torch.maximum(ay2, by2) - torch.minimum(ay1, by1)).clamp_min(0.0)
    c_area = ew * eh
    c2 = ew * ew + eh * eh

    # GIoU = IoU - (C面积 - union)/C面积
    giou = iou - _safe_div(c_area - union, c_area, c_area > 0)

    # DIoU = IoU - rho^2 / c^2
    rho2 = ((ax1 + ax2 - bx1 - bx2) * 0.5) ** 2 + ((ay1 + ay2 - by1 - by2) * 0.5) ** 2
    diou = iou - _safe_div(rho2, c2, c2 > 0)

    # CIoU = DIoU - alpha * v
    aw = (ax2 - ax1).clamp_min(1e-7)
    ah = (ay2 - ay1).clamp_min(1e-7)
    bw = (bx2 - bx1).clamp_min(1e-7)
    bh = (by2 - by1).clamp_min(1e-7)
    v = (4.0 / (math.pi ** 2)) * (torch.atan(bw / bh) - torch.atan(aw / ah)) ** 2
    den = (1.0 - iou) + v
    alpha = _safe_div(v, den, den > 0)
    ciou = diou - alpha * v

    loss_giou, loss_diou, loss_ciou = 1.0 - giou, 1.0 - diou, 1.0 - ciou
    if reduction == "none":
        return (loss_giou, loss_diou, loss_ciou)
    if reduction == "mean":
        if pred.shape[0] == 0:
            zero = torch.zeros((), dtype=pred.dtype)
            return (zero, zero, zero)
        return (loss_giou.mean(), loss_diou.mean(), loss_ciou.mean())
    raise ValueError(f"unknown reduction: {reduction!r}")
