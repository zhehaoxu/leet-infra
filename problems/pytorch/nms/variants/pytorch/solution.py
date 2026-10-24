"""参考答案：非极大值抑制 NMS（目标框去重的贪心算法）。

boxes (N,4) 采用 [x1,y1,x2,y2]；scores (N,)。步骤：
1. 按分数降序排序；
2. 取当前最高分框保留，把与它 IoU > 阈值的其余框全部抑制（丢弃）；
3. 重复直到无候选。返回保留框的原始索引，按分数从高到低排列。
"""

import torch


def _iou_matrix(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    xx1 = torch.maximum(x1[:, None], x1[None, :])
    yy1 = torch.maximum(y1[:, None], y1[None, :])
    xx2 = torch.minimum(x2[:, None], x2[None, :])
    yy2 = torch.minimum(y2[:, None], y2[None, :])
    inter = (xx2 - xx1).clamp_min(0) * (yy2 - yy1).clamp_min(0)
    union = area[:, None] + area[None, :] - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(union))


def solve(boxes, scores, iou_thresh):
    iou = _iou_matrix(boxes)
    order = scores.argsort(descending=True).tolist()
    keep = []
    while order:
        i0 = order.pop(0)
        keep.append(i0)
        # 抑制与 i0 的 IoU > 阈值的候选（保持原始相对顺序）
        order = [j for j in order if iou[i0, j].item() <= iou_thresh]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
