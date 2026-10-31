"""DIoU-NMS 抑制判据账（solution）。

按 1911.08287 公式：DIoU = IoU − ρ²/c²。
"""

import math


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def diou(a, b):
    ca, cb = _center(a), _center(b)
    rho2 = (ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2
    ex1, ey1 = min(a[0], b[0]), min(a[1], b[1])
    ex2, ey2 = max(a[2], b[2]), max(a[3], b[3])
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2
    return iou(a, b) - rho2 / c2 if c2 > 0 else iou(a, b)


def solve(boxes, scores, thresh):
    if not isinstance(boxes, list) or not boxes:
        raise ValueError("boxes must be a non-empty list")
    if len(scores) != len(boxes):
        raise ValueError("scores length must match boxes")
    for b in boxes:
        if not isinstance(b, list) or len(b) != 4 or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in b
        ):
            raise ValueError("each box must be [x1,y1,x2,y2] finite floats")
        if b[2] < b[0] or b[3] < b[1]:
            raise ValueError("box must satisfy x2>=x1, y2>=y1")
    if not all(isinstance(s, (int, float)) and math.isfinite(s)
               for s in scores):
        raise ValueError("scores must be finite floats")
    if not isinstance(thresh, (int, float)) or not math.isfinite(thresh) \
            or not 0 <= thresh <= 1:
        raise ValueError("thresh must be in [0, 1]")

    order = sorted(range(len(boxes)), key=lambda i: (-scores[i], i))
    keep = []
    suppressed = {}
    alive = list(order)
    while alive:
        cur = alive[0]
        keep.append(cur)
        rest = []
        for j in alive[1:]:
            if diou(boxes[cur], boxes[j]) > thresh:   # 严格大于
                suppressed[j] = cur
            else:
                rest.append(j)
        alive = rest
    return {"keep": keep, "suppressed_by": suppressed}
