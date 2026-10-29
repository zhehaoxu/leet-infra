"""参考答案：旋转框 IoU（Sutherland–Hodgman 裁剪 + 鞋带公式）与贪心 NMS。"""

import torch


def _corners(boxes):
    """(N,5) -> (N,4,2) 的逆时针四角，float64。"""
    b = boxes.to(torch.float64)
    cx, cy, w, h, th = b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4]
    hw, hh = w / 2.0, h / 2.0
    xs = torch.stack([hw, hw, -hw, -hw], dim=1)
    ys = torch.stack([-hh, hh, hh, -hh], dim=1)
    c, s = torch.cos(th).unsqueeze(1), torch.sin(th).unsqueeze(1)
    px = xs * c - ys * s + cx.unsqueeze(1)
    py = xs * s + ys * c + cy.unsqueeze(1)
    return torch.stack([px, py], dim=2)


def _area(poly):
    """鞋带公式，(V,2) -> float。"""
    if poly.shape[0] < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(torch.dot(x, torch.roll(y, -1)) - torch.dot(y, torch.roll(x, -1))) / 2.0)


def _clip(subject, clip):
    """用逆时针凸多边形 clip 裁 subject（Sutherland–Hodgman）。"""
    out = subject
    n_clip = clip.shape[0]
    for i in range(n_clip):
        a = clip[i]
        b = clip[(i + 1) % n_clip]
        ex, ey = float(b[0] - a[0]), float(b[1] - a[1])

        def side(p, ax=float(a[0]), ay=float(a[1]), ex=ex, ey=ey):
            return ex * (float(p[1]) - ay) - ey * (float(p[0]) - ax)

        m = out.shape[0]
        kept = []
        for k in range(m):
            cur = out[k]
            prev = out[(k - 1) % m]
            sc, sp = side(cur), side(prev)
            if sc >= 0.0:
                if sp < 0.0:
                    t = sp / (sp - sc)
                    kept.append(prev + (cur - prev) * t)
                kept.append(cur)
            elif sp >= 0.0:
                t = sp / (sp - sc)
                kept.append(prev + (cur - prev) * t)
        if not kept:
            return torch.zeros(0, 2, dtype=torch.float64)
        out = torch.stack(kept)
    return out


def _iou_pair(pa, pb, area_a, area_b):
    inter = _area(_clip(pa, pb))
    if inter <= 0.0:
        return 0.0
    inter = min(inter, area_a, area_b)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def solve(boxes, scores, iou_thr=0.5):
    if not isinstance(boxes, torch.Tensor) or not isinstance(scores, torch.Tensor):
        raise ValueError("boxes 与 scores 必须是 torch.Tensor")
    if boxes.dim() != 2 or boxes.shape[1] != 5:
        raise ValueError(f"boxes 必须是 (N,5)，got {tuple(boxes.shape)}")
    if scores.dim() != 1:
        raise ValueError(f"scores 必须是一维，got {tuple(scores.shape)}")
    if scores.shape[0] != boxes.shape[0]:
        raise ValueError(
            f"scores 长度必须等于 boxes 行数 {boxes.shape[0]}，got {scores.shape[0]}")
    if boxes.dtype not in (torch.float32, torch.float64) or boxes.dtype != scores.dtype:
        raise ValueError(f"boxes/scores 必须是同一种 float32 或 float64，got {boxes.dtype}/{scores.dtype}")
    if not bool(torch.isfinite(boxes).all()) or not bool(torch.isfinite(scores).all()):
        raise ValueError("boxes/scores 不能含 NaN 或 Inf")
    if bool((boxes[:, 2] <= 0).any()) or bool((boxes[:, 3] <= 0).any()):
        raise ValueError("w 与 h 必须 > 0")
    thr = float(iou_thr)
    if not (0.0 <= thr <= 1.0):
        raise ValueError(f"iou_thr 必须在 [0,1] 内，got {thr}")

    n = int(boxes.shape[0])
    order = torch.argsort(-scores.to(torch.float64), stable=True).tolist()

    if n == 0:
        return (torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.bool),
                torch.zeros(0, 0, dtype=torch.float64))

    corners = _corners(boxes)
    areas = [_area(corners[i]) for i in range(n)]
    iou = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        iou[i, i] = 1.0
        for j in range(i + 1, n):
            v = _iou_pair(corners[i], corners[j], areas[i], areas[j])
            iou[i, j] = v
            iou[j, i] = v

    kept = []
    for idx in order:
        if all(float(iou[idx, k]) <= thr for k in kept):
            kept.append(idx)

    keep = torch.tensor(kept, dtype=torch.int64)
    mask = torch.zeros(n, dtype=torch.bool)
    if kept:
        mask[keep] = True
    return keep, mask, iou
