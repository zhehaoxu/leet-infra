import torch


def _iou_matrix(a, b):
    """(P,4) x (G,4) -> (P,G) float64 IoU，零面积记 0。"""
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


def _ap(prec, rec, method):
    n = len(prec)
    if n == 0:
        return 0.0
    if method == "all":
        env = [0.0] * n
        best = 0.0
        for i in range(n - 1, -1, -1):
            best = max(best, prec[i])
            env[i] = best
        ap, prev = 0.0, 0.0
        for i in range(n):
            ap += (rec[i] - prev) * env[i]
            prev = rec[i]
        return ap
    grid = 101 if method == "coco" else 11
    total = 0.0
    for k in range(grid):
        r = k / (grid - 1)
        p = 0.0
        for i in range(n):
            if rec[i] >= r and prec[i] > p:
                p = prec[i]
        total += p
    return total / grid


def solve(pred_boxes, pred_scores, pred_cls, pred_img, gt_boxes, gt_cls, gt_img,
          iou_thrs=(0.5,), method="coco"):
    for name, t in (("pred_boxes", pred_boxes), ("gt_boxes", gt_boxes)):
        if not isinstance(t, torch.Tensor) or t.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name} 必须是 float32/float64 张量")
        if t.dim() != 2 or t.shape[1] != 4:
            raise ValueError(f"{name} 形状必须是 (N,4)")
    if pred_boxes.dtype != gt_boxes.dtype:
        raise ValueError("pred_boxes 与 gt_boxes 必须同 dtype")
    if not isinstance(pred_scores, torch.Tensor) or pred_scores.dtype != pred_boxes.dtype:
        raise ValueError("pred_scores 必须与 pred_boxes 同 dtype")
    if pred_scores.dim() != 1 or pred_scores.shape[0] != pred_boxes.shape[0]:
        raise ValueError("pred_scores 形状必须是 (P,)")
    for name, t, n in (("pred_cls", pred_cls, pred_boxes.shape[0]),
                       ("pred_img", pred_img, pred_boxes.shape[0]),
                       ("gt_cls", gt_cls, gt_boxes.shape[0]),
                       ("gt_img", gt_img, gt_boxes.shape[0])):
        if not isinstance(t, torch.Tensor) or t.dtype not in (torch.int64, torch.int32):
            raise ValueError(f"{name} 必须是整型张量")
        if t.dim() != 1 or t.shape[0] != n:
            raise ValueError(f"{name} 形状必须是 ({n},)")
    if len(iou_thrs) == 0:
        raise ValueError("iou_thrs 不能为空")
    if any(not (0.0 <= float(v) <= 1.0) for v in iou_thrs):
        raise ValueError("iou_thrs 元素必须在 [0,1] 内")
    if method not in ("all", "11point", "coco"):
        raise ValueError(f"未知 method: {method}")

    classes = sorted(set(int(v) for v in gt_cls.tolist()))
    T = len(iou_thrs)
    ap = torch.zeros(T, len(classes), dtype=torch.float64)
    if len(classes) == 0:
        return ap, torch.zeros(T, dtype=torch.float64)
    if pred_boxes.shape[0] == 0:
        return ap, ap.mean(dim=1)

    iou = _iou_matrix(pred_boxes, gt_boxes)
    pc = [int(v) for v in pred_cls.tolist()]
    pi = [int(v) for v in pred_img.tolist()]
    gc = [int(v) for v in gt_cls.tolist()]
    gi = [int(v) for v in gt_img.tolist()]
    order = torch.argsort(-pred_scores, stable=True).tolist()

    for ti, thr in enumerate(iou_thrs):
        thr = float(thr)
        for ci, cls in enumerate(classes):
            preds = [p for p in order if pc[p] == cls]
            gts = [g for g in range(len(gc)) if gc[g] == cls]
            n_gt = len(gts)
            used = [False] * len(gc)
            prec, rec = [], []
            tp, fp = 0, 0
            for p in preds:
                best_v, best_g = -1.0, -1
                for g in gts:
                    if used[g] or gi[g] != pi[p]:
                        continue
                    v = float(iou[p, g])
                    if v > best_v:
                        best_v, best_g = v, g
                if best_g >= 0 and best_v >= thr:
                    used[best_g] = True
                    tp += 1
                else:
                    fp += 1
                prec.append(tp / (tp + fp))
                rec.append(tp / n_gt)
            ap[ti, ci] = _ap(prec, rec, method)

    mean_ap = ap.mean(dim=1) if len(classes) else torch.zeros(T, dtype=torch.float64)
    return ap, mean_ap
