"""参考答案：交并比 IoU（逐对，box 采用 [x1, y1, x2, y2] 左上/右下角坐标）。

交集 = 两 box 在 x、y 方向重叠区（宽高用 clamp_min(0) 防负）；并集 = 面积和 - 交集。
IoU = 交集 / 并集。支持广播：a、b 形状 (...,4)。
"""

import torch


def solve(a, b):
    ax1, ay1, ax2, ay2 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx1, by1, bx2, by2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    ix1 = torch.maximum(ax1, bx1)
    iy1 = torch.maximum(ay1, by1)
    ix2 = torch.minimum(ax2, bx2)
    iy2 = torch.minimum(ay2, by2)
    iw = (ix2 - ix1).clamp_min(0)
    ih = (iy2 - iy1).clamp_min(0)
    inter = iw * ih

    area_a = ((ax2 - ax1).clamp_min(0)) * ((ay2 - ay1).clamp_min(0))
    area_b = ((bx2 - bx1).clamp_min(0)) * ((by2 - by1).clamp_min(0))
    union = area_a + area_b - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
    return iou
