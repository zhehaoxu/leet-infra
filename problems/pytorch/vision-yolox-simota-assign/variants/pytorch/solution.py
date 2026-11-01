"""YOLOX SimOTA dynamic label assignment, official get_assignments path.

Reproduced from Megvii-BaseDetection/YOLOX `yolox/models/yolo_head.py`
(get_geometry_constraint + cost + simota_matching) and
`yolox/utils/boxes.py::bboxes_iou` (xyxy=False, cxcywh).

Pipeline:
  1. geometry constraint: anchor center (x_shift+0.5)*stride must lie
     strictly inside [gt - 1.5*stride, gt + 1.5*stride] per axis
     (center_radius=1.5, dist scales with the anchor's own stride);
     anchor_filter = anchors inside ANY gt's neighborhood
  2. candidates = anchors passing the filter; IoU of gt vs candidate
     predicted boxes in cxcywh (touching edges do NOT count: tl<br strict)
  3. cost = cls_bce_sum + 3.0 * (-log(iou + 1e-8)) + 1e6 * (~geometry)
     where cls score = sqrt(sigmoid(cls) * sigmoid(obj)) per class
  4. dynamic_k per gt = max(int(sum(top-10 ious)), 1); pick that many
     lowest-cost anchors per gt
  5. an anchor matched by multiple gts keeps only its cheapest gt
  6. outputs: per-anchor final fg mask, matched gt index/class, the
     matched IoU value, and the full cost/matching ledgers
"""
import math


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def solve(gt_bboxes, gt_classes, expanded_strides, x_shifts, y_shifts,
          cls_preds, obj_preds, bboxes_preds, num_classes):
    num_gt = len(gt_bboxes)
    num_anchor = len(expanded_strides)
    if num_gt == 0:
        raise ValueError("need at least one gt")
    if num_anchor == 0:
        raise ValueError("need at least one anchor")
    if len(gt_classes) != num_gt or len(bboxes_preds) != num_anchor \
            or len(cls_preds) != num_anchor or len(obj_preds) != num_anchor \
            or len(x_shifts) != num_anchor or len(y_shifts) != num_anchor:
        raise ValueError("input length mismatch")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    for row in gt_bboxes + bboxes_preds:
        if len(row) != 4 or not all(math.isfinite(v) for v in row):
            raise ValueError("bbox rows must be 4 finite values")
        if row[2] <= 0 or row[3] <= 0:
            raise ValueError("box w/h must be positive")
    for c in gt_classes:
        if not (0 <= c < num_classes):
            raise ValueError("gt class out of range")
    for row in cls_preds:
        if len(row) != num_classes:
            raise ValueError("cls row length must equal num_classes")
    for s in expanded_strides:
        if s < 1:
            raise ValueError("stride must be >= 1")

    strides = [float(s) for s in expanded_strides]
    x_centers = [(x_shifts[k] + 0.5) * strides[k] for k in range(num_anchor)]
    y_centers = [(y_shifts[k] + 0.5) * strides[k] for k in range(num_anchor)]

    # 1. geometry constraint (per anchor stride-scaled 1.5 radius)
    is_in_centers = []
    for g in range(num_gt):
        gx, gy = gt_bboxes[g][0], gt_bboxes[g][1]
        row = []
        for k in range(num_anchor):
            dist = strides[k] * 1.5
            c_l = x_centers[k] - (gx - dist)
            c_r = (gx + dist) - x_centers[k]
            c_t = y_centers[k] - (gy - dist)
            c_b = (gy + dist) - y_centers[k]
            row.append(min(c_l, c_t, c_r, c_b) > 0.0)
        is_in_centers.append(row)
    anchor_filter = [any(is_in_centers[g][k] for g in range(num_gt))
                     for k in range(num_anchor)]
    cand = [k for k in range(num_anchor) if anchor_filter[k]]
    n_cand = len(cand)

    def _iou_cxcywh(a, b):
        # xyxy=False path: touching edges (tl == br) give en=0 -> IoU 0
        ax1, ay1 = a[0] - a[2] / 2.0, a[1] - a[3] / 2.0
        ax2, ay2 = a[0] + a[2] / 2.0, a[1] + a[3] / 2.0
        bx1, by1 = b[0] - b[2] / 2.0, b[1] - b[3] / 2.0
        bx2, by2 = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
        tlx, tly = max(ax1, bx1), max(ay1, by1)
        brx, bry = min(ax2, bx2), min(ay2, by2)
        en = (tlx < brx) and (tly < bry)
        if not en:
            return 0.0
        area_i = (brx - tlx) * (bry - tly)
        return area_i / (a[2] * a[3] + b[2] * b[3] - area_i)

    pair_ious = [[_iou_cxcywh(gt_bboxes[g], bboxes_preds[k])
                  for k in cand] for g in range(num_gt)]

    # 3. cost
    cost = []
    for g in range(num_gt):
        row = []
        for j, k in enumerate(cand):
            cls_row = cls_preds[k]
            obj_sig = _sig(obj_preds[k])
            cls_loss = 0.0
            for c in range(num_classes):
                p = math.sqrt(_sig(cls_row[c]) * obj_sig)
                y = 1.0 if c == gt_classes[g] else 0.0
                cls_loss -= (y * math.log(p) + (1 - y) * math.log(1.0 - p))
            iou_loss = -math.log(pair_ious[g][j] + 1e-8)
            geo = 0.0 if is_in_centers[g][k] else 1e6
            row.append(cls_loss + 3.0 * iou_loss + geo)
        cost.append(row)

    # 4. dynamic k
    n_candidate_k = min(10, n_cand)
    dynamic_ks = []
    for g in range(num_gt):
        top_ious = sorted(pair_ious[g], reverse=True)[:n_candidate_k]
        dynamic_ks.append(max(int(sum(top_ious)), 1))

    # 5. per-gt lowest-cost picks, then conflict arbitration
    matching = [[0] * n_cand for _ in range(num_gt)]
    for g in range(num_gt):
        order = sorted(range(n_cand), key=lambda j: (cost[g][j], j))
        for j in order[:dynamic_ks[g]]:
            matching[g][j] = 1

    anchor_matching_gt = [sum(matching[g][j] for g in range(num_gt))
                          for j in range(n_cand)]
    for j in range(n_cand):
        if anchor_matching_gt[j] > 1:
            best_g = min(range(num_gt), key=lambda g: (cost[g][j], g))
            for g in range(num_gt):
                matching[g][j] = 0
            matching[best_g][j] = 1

    fg_mask_inboxes = [any(matching[g][j] for g in range(num_gt))
                       for j in range(n_cand)]
    num_fg = sum(fg_mask_inboxes)

    matched_gt_inds = [-1] * num_anchor
    gt_matched_classes = [-1] * num_anchor
    pred_ious = [0.0] * num_anchor
    for j, k in enumerate(cand):
        if not fg_mask_inboxes[j]:
            continue
        g = next(gg for gg in range(num_gt) if matching[gg][j] == 1)
        matched_gt_inds[k] = g
        gt_matched_classes[k] = gt_classes[g]
        pred_ious[k] = pair_ious[g][j]

    fg_mask = [False] * num_anchor
    for j, k in enumerate(cand):
        fg_mask[k] = fg_mask_inboxes[j]

    return {
        "anchor_filter": anchor_filter,
        "is_in_centers": is_in_centers,
        "ious": pair_ious,
        "cost": cost,
        "dynamic_ks": dynamic_ks,
        "matching_matrix": matching,
        "fg_mask": fg_mask,
        "num_fg": num_fg,
        "matched_gt_inds": matched_gt_inds,
        "gt_matched_classes": gt_matched_classes,
        "pred_ious_this_matching": pred_ious,
    }
