"""YOLOv8 TAL (Task-Aligned Assigner), official ultralytics path.

Reproduced from ultralytics/ultralytics `utils/tal.py::TaskAlignedAssigner`
(single image, all GTs valid, topk2 == topk so the secondary filter is off).

Pipeline:
  1. center-in-gt: anchor center strictly inside the GT box (eps margin);
     GT sides smaller than stride_val are enlarged about their center
  2. bbox_scores = pd_scores[anchor, gt_class] (sigmoid-applied scores)
  3. overlap = CIoU(gt, pd) clamped to 0; align = score^alpha * overlap^beta
  4. top-k per GT by align metric (topk_mask all-true for valid GTs)
  5. mask_pos = topk_mask * in-gts mask
  6. multi-gt arbitration: an anchor claimed by several GTs keeps the one
     with the highest overlap (first max on ties)
  7. normalization: align *= mask_pos_arb;
     pos_align = max over anchors per gt; pos_overlap = max overlap per gt;
     align = align * pos_overlap / (pos_align + eps);
     norm = max over gts per anchor;
     target_scores = one_hot(gt_class) * norm  (soft label on the gt class)

CIoU follows the standard closed form (no +1 pixel convention), xyxy boxes.
"""
import math


def _ciou(gt, pd):
    ix1, iy1 = max(gt[0], pd[0]), max(gt[1], pd[1])
    ix2, iy2 = min(gt[2], pd[2]), min(gt[3], pd[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_g = (gt[2] - gt[0]) * (gt[3] - gt[1])
    area_p = (pd[2] - pd[0]) * (pd[3] - pd[1])
    union = area_g + area_p - inter
    iou = inter / union if union > 0 else 0.0
    gcx, gcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    pcx, pcy = (pd[0] + pd[2]) / 2, (pd[1] + pd[3]) / 2
    rho2 = (gcx - pcx) ** 2 + (gcy - pcy) ** 2
    cw = max(gt[2], pd[2]) - min(gt[0], pd[0])
    ch = max(gt[3], pd[3]) - min(gt[1], pd[1])
    c2 = cw ** 2 + ch ** 2
    gw, gh = gt[2] - gt[0], gt[3] - gt[1]
    pw, ph = pd[2] - pd[0], pd[3] - pd[1]
    v = (4.0 / math.pi ** 2) * (math.atan(pw / ph) - math.atan(gw / gh)) ** 2
    denom = 1.0 - iou + v
    alpha = v / denom if denom > 0 else 0.0
    return max(0.0, iou - rho2 / c2 - alpha * v)


def solve(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes,
          num_classes, topk=13, alpha=1.0, beta=6.0, stride_val=16.0,
          eps=1e-9):
    G = len(gt_bboxes)
    A = len(pd_bboxes)
    if G == 0:
        raise ValueError("need at least one gt")
    if A == 0:
        raise ValueError("need at least one anchor")
    if A < topk:
        raise ValueError("anchors must be >= topk (official would raise)")
    if topk < 1:
        raise ValueError("topk must be positive")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    if not (alpha >= 0 and beta >= 0):
        raise ValueError("alpha/beta must be non-negative")
    if not (stride_val > 0) or not (eps > 0):
        raise ValueError("stride_val/eps must be positive")
    for row in pd_scores:
        if len(row) != num_classes:
            raise ValueError("pd_scores row length must equal num_classes")
        if any(v < 0 or v > 1 or not math.isfinite(v) for v in row):
            raise ValueError("pd_scores must be probabilities in [0,1]")
    for row in pd_bboxes + gt_bboxes:
        if len(row) != 4 or not all(math.isfinite(v) for v in row):
            raise ValueError("bbox rows must be 4 finite values")
        if row[2] <= row[0] or row[3] <= row[1]:
            raise ValueError("boxes must be x1<x2, y1<y2")
    for c in gt_labels:
        if not (0 <= c < num_classes):
            raise ValueError("gt class out of range")
    for p in anc_points:
        if len(p) != 2 or not all(math.isfinite(v) for v in p):
            raise ValueError("anchor points must be [x, y]")

    # 1. center-in-gt with small-side enlargement
    expanded = []
    for g in range(G):
        x1, y1, x2, y2 = gt_bboxes[g]
        w, h = x2 - x1, y2 - y1
        if w < stride_val:
            cx = (x1 + x2) / 2
            x1, x2 = cx - stride_val / 2, cx + stride_val / 2
        if h < stride_val:
            cy = (y1 + y2) / 2
            y1, y2 = cy - stride_val / 2, cy + stride_val / 2
        expanded.append([x1, y1, x2, y2])
    mask_in_gts = []
    for g in range(G):
        x1, y1, x2, y2 = expanded[g]
        row = []
        for k in range(A):
            ax, ay = anc_points[k]
            row.append(ax - x1 > eps and ay - y1 > eps
                       and x2 - ax > eps and y2 - ay > eps)
        mask_in_gts.append(row)

    # 2-3. scores / overlaps / align metric
    bbox_scores = []
    overlaps = []
    align_metric = []
    for g in range(G):
        srow, orow, arow = [], [], []
        for k in range(A):
            s = pd_scores[k][gt_labels[g]] if mask_in_gts[g][k] else 0.0
            o = _ciou(gt_bboxes[g], pd_bboxes[k]) if mask_in_gts[g][k] \
                else 0.0
            srow.append(s)
            orow.append(o)
            arow.append(s ** alpha * o ** beta if mask_in_gts[g][k] else 0.0)
        bbox_scores.append(srow)
        overlaps.append(orow)
        align_metric.append(arow)

    # 4. top-k per GT (ties: smaller anchor index first)
    mask_topk = [[0] * A for _ in range(G)]
    for g in range(G):
        order = sorted(range(A), key=lambda k: (-align_metric[g][k], k))
        for k in order[:topk]:
            mask_topk[g][k] = 1

    # 5. merge
    mask_pos = [[mask_topk[g][k] and mask_in_gts[g][k] for k in range(A)]
                for g in range(G)]

    # 6. multi-gt arbitration by highest overlap
    mask_pos_pre = [row[:] for row in mask_pos]
    for k in range(A):
        claim = [g for g in range(G) if mask_pos[g][k]]
        if len(claim) > 1:
            best = max(claim, key=lambda g: (overlaps[g][k], -g))
            for g in claim:
                mask_pos[g][k] = 0
            mask_pos[best][k] = 1

    fg_mask = [any(mask_pos[g][k] for g in range(G)) for k in range(A)]
    target_gt_idx = [-1] * A
    for k in range(A):
        if fg_mask[k]:
            target_gt_idx[k] = next(g for g in range(G) if mask_pos[g][k])

    # 7. normalization
    pos_align = [max(align_metric[g][k] for k in range(A)
                     if mask_pos[g][k]) if any(mask_pos[g]) else 0.0
                 for g in range(G)]
    pos_over = [max(overlaps[g][k] for k in range(A)
                    if mask_pos[g][k]) if any(mask_pos[g]) else 0.0
                for g in range(G)]
    norm_align = []
    for k in range(A):
        best = 0.0
        for g in range(G):
            if mask_pos[g][k]:
                v = align_metric[g][k] * pos_over[g] / (pos_align[g] + eps)
                best = max(best, v)
        norm_align.append(best)
    target_scores = []
    for k in range(A):
        row = [0.0] * num_classes
        if fg_mask[k]:
            row[gt_labels[target_gt_idx[k]]] = norm_align[k]
        target_scores.append(row)

    return {
        "expanded_boxes": expanded,
        "mask_in_gts": mask_in_gts,
        "bbox_scores": bbox_scores,
        "overlaps": overlaps,
        "align_metric": align_metric,
        "mask_topk": mask_topk,
        "mask_pos": mask_pos_pre,
        "mask_pos_arb": mask_pos,
        "fg_mask": fg_mask,
        "target_gt_idx": target_gt_idx,
        "pos_align_metrics": pos_align,
        "pos_overlaps": pos_over,
        "norm_align_metric": norm_align,
        "target_scores": target_scores,
    }
