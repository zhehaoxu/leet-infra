"""参考答案：Soft-NMS 置信度连续衰减（迭代选择 + 向量化衰减）。

口径：IoU 同 iou-compute（交集宽高 clamp_min(0)，并集为 0 记 IoU=0）；
linear 仅在 IoU > iou_thr 时乘 (1 − IoU)，gaussian 无条件乘 exp(−IoU²/sigma)；
每轮选择后删除衰减后分数 < score_thr 的剩余框，并列最高分取最小原始下标。
"""

import torch


def _iou_matrix(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    iw = (torch.minimum(x2[:, None], x2[None, :])
          - torch.maximum(x1[:, None], x1[None, :])).clamp_min(0.0)
    ih = (torch.minimum(y2[:, None], y2[None, :])
          - torch.maximum(y1[:, None], y1[None, :])).clamp_min(0.0)
    inter = iw * ih
    union = area[:, None] + area[None, :] - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


def solve(boxes, scores, method="linear", sigma=0.5, iou_thr=0.3, score_thr=0.001):
    if method not in ("linear", "gaussian"):
        raise ValueError(f"unknown method: {method!r}")
    if method == "gaussian" and sigma <= 0:
        raise ValueError("sigma must be > 0 when method='gaussian'")

    n = boxes.shape[0]
    if n == 0:
        return (torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=scores.dtype))

    iou = _iou_matrix(boxes)
    cur = scores.clone()
    active = torch.ones(n, dtype=torch.bool)
    keep_idx, keep_score = [], []
    while bool(active.any()):
        # 无效位置填 -inf，argmax 并列时返回首个（即原始下标最小者）
        masked = torch.where(active, cur, torch.full_like(cur, float("-inf")))
        best = int(masked.argmax())
        keep_idx.append(best)
        keep_score.append(cur[best].clone())
        active[best] = False
        row = iou[best]
        if method == "linear":
            factor = torch.where(row > iou_thr, 1.0 - row, torch.ones_like(row))
        else:
            factor = torch.exp(-(row * row) / sigma)
        # 只衰减仍活跃的框；已选中的与已删除的分数保持不变
        cur = torch.where(active, cur * factor, cur)
        active &= cur >= score_thr
    return torch.tensor(keep_idx, dtype=torch.long), torch.stack(keep_score)
