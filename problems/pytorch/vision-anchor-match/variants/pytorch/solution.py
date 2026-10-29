import math

import torch


def _anchors_for(dtype, feature_sizes, strides, scales, ratios, image_size):
    w_img, h_img = float(image_size[0]), float(image_size[1])
    parts = []
    for (h, w), stride in zip(feature_sizes, strides):
        ys = (torch.arange(h, dtype=dtype) + 0.5) * float(stride)
        xs = (torch.arange(w, dtype=dtype) + 0.5) * float(stride)
        cy, cx = torch.meshgrid(ys, xs, indexing="ij")
        combos = []
        for s in scales:
            for r in ratios:
                half_w = 0.5 * float(s) * math.sqrt(float(r))
                half_h = 0.5 * float(s) / math.sqrt(float(r))
                x1 = (cx - half_w).clamp(0.0, w_img)
                y1 = (cy - half_h).clamp(0.0, h_img)
                x2 = (cx + half_w).clamp(0.0, w_img)
                y2 = (cy + half_h).clamp(0.0, h_img)
                combos.append(torch.stack([x1, y1, x2, y2], dim=-1))
        parts.append(torch.stack(combos, dim=2).reshape(-1, 4))  # (H,W,S*R,4) 格优先
    return torch.cat(parts, dim=0) if parts else torch.zeros(0, 4, dtype=dtype)


def _iou_matrix(a, b):
    """(A,4) x (G,4) -> (A,G)，裁剪坐标、零面积记 0。"""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


def solve(gt_boxes, feature_sizes, strides, scales, ratios, image_size,
          pos_thr=0.5, neg_thr=0.4):
    if not isinstance(gt_boxes, torch.Tensor) or gt_boxes.dtype not in (torch.float32, torch.float64):
        raise ValueError("gt_boxes 必须是 float32/float64 张量")
    if gt_boxes.dim() != 2 or gt_boxes.shape[1] != 4:
        raise ValueError(f"gt_boxes 形状必须是 (G,4)，got {tuple(gt_boxes.shape)}")
    if len(feature_sizes) != len(strides) or len(feature_sizes) == 0:
        raise ValueError("feature_sizes 与 strides 必须等长且非空")
    if any(int(h) <= 0 or int(w) <= 0 for h, w in feature_sizes):
        raise ValueError("feature_sizes 各项必须为正整数")
    if any(int(s) <= 0 for s in strides):
        raise ValueError("strides 各项必须为正整数")
    if not scales or not ratios or any(float(s) <= 0 for s in scales) \
            or any(float(r) <= 0 for r in ratios):
        raise ValueError("scales / ratios 必须非空且为正")
    if len(image_size) != 2 or int(image_size[0]) <= 0 or int(image_size[1]) <= 0:
        raise ValueError("image_size 必须是两个正整数 (W,H)")
    if not (0.0 <= float(neg_thr) <= float(pos_thr)):
        raise ValueError("要求 0 <= neg_thr <= pos_thr")

    anchors = _anchors_for(gt_boxes.dtype, feature_sizes, strides, scales, ratios, image_size)
    n = anchors.shape[0]
    labels = torch.full((n,), -1, dtype=torch.int64)
    if gt_boxes.shape[0] == 0:
        counts = torch.tensor([0, n, 0], dtype=torch.int64)
        return anchors, labels, counts

    iou = _iou_matrix(anchors, gt_boxes)
    best_iou, best_gt = iou.max(dim=1)  # 并列取列下标最小者

    labels = torch.where(best_iou >= float(pos_thr), best_gt, labels)
    labels = torch.where(best_iou < float(neg_thr), torch.full_like(labels, -1), labels)
    mid = (best_iou >= float(neg_thr)) & (best_iou < float(pos_thr))
    labels = torch.where(mid, torch.full_like(labels, -2), labels)

    # 保底：每个 GT 取 best_gt 指向它的 anchor 中 IoU 最大者（并列取小下标）
    for g in range(gt_boxes.shape[0]):
        cand = (best_gt == g).nonzero(as_tuple=False).flatten()
        if cand.numel() == 0:
            continue
        vals = best_iou[cand]
        pick = cand[int(vals.argmax())]
        labels[pick] = g

    pos = int((labels >= 0).sum())
    neg = int((labels == -1).sum())
    ign = int((labels == -2).sum())
    return anchors, labels, torch.tensor([pos, neg, ign], dtype=torch.int64)
