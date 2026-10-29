import torch


def _iou(a, b):
    """单对 xyxy 框的 IoU（float64 标量），面积都取 clip 后的正值。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    inter = max(iw, 0.0) * max(ih, 0.0)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _fuse(members, num_models, dtype):
    """members: [(box_list, score, label)]，返回该簇的融合框与融合置信度。"""
    w = torch.tensor([m[1] for m in members], dtype=torch.float64)
    bs = torch.tensor([m[0] for m in members], dtype=torch.float64)
    if float(w.sum()) > 0:
        box = (bs * w[:, None]).sum(dim=0) / w.sum()
    else:
        box = bs.mean(dim=0)
    n = len(members)
    score = float(w.mean()) * min(n, num_models) / num_models
    return box.to(dtype), torch.tensor(score, dtype=dtype)


def solve(boxes, scores, labels, iou_thr=0.55, num_models=1):
    if not isinstance(boxes, torch.Tensor) or boxes.dtype not in (torch.float32, torch.float64):
        raise ValueError("boxes 必须是 float32/float64 张量")
    if boxes.dim() != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes 形状必须是 (N,4)，got {tuple(boxes.shape)}")
    if not isinstance(scores, torch.Tensor) or scores.dtype != boxes.dtype:
        raise ValueError("scores 必须是张量且与 boxes 同 dtype")
    if scores.dim() != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError("scores 形状必须是 (N,)")
    if not isinstance(labels, torch.Tensor) or labels.dtype not in (torch.int64, torch.int32):
        raise ValueError("labels 必须是整型张量")
    if labels.dim() != 1 or labels.shape[0] != boxes.shape[0]:
        raise ValueError("labels 形状必须是 (N,)")
    if not (0.0 <= float(iou_thr) <= 1.0):
        raise ValueError("iou_thr 必须在 [0,1] 内")
    if int(num_models) != num_models or int(num_models) < 1:
        raise ValueError("num_models 必须是 >=1 的整数")

    dtype = boxes.dtype
    n = boxes.shape[0]
    if n == 0:
        return (torch.zeros(0, 4, dtype=dtype), torch.zeros(0, dtype=dtype),
                torch.zeros(0, dtype=torch.int64))

    order = torch.argsort(-scores, stable=True).tolist()
    bl = boxes.to(torch.float64).tolist()
    sl = scores.to(torch.float64).tolist()
    ll = labels.tolist()

    clusters = []  # [{"label", "members", "box"}]
    for i in order:
        lab = int(ll[i])
        best_j, best_iou = -1, float(iou_thr)
        for j, c in enumerate(clusters):
            if c["label"] != lab:
                continue
            v = _iou(bl[i], c["box"])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            c = clusters[best_j]
            c["members"].append((bl[i], sl[i], lab))
            c["box"] = _fuse(c["members"], int(num_models), torch.float64)[0].tolist()
        else:
            clusters.append({"label": lab, "members": [(bl[i], sl[i], lab)], "box": list(bl[i])})

    out_boxes, out_scores, out_labels = [], [], []
    for c in clusters:
        box, score = _fuse(c["members"], int(num_models), torch.float64)
        out_boxes.append(box)
        out_scores.append(score)
        out_labels.append(c["label"])
    fb = torch.stack(out_boxes).to(dtype) if out_boxes else torch.zeros(0, 4, dtype=dtype)
    fs = torch.stack(out_scores).to(dtype) if out_scores else torch.zeros(0, dtype=dtype)
    fl = torch.tensor(out_labels, dtype=torch.int64)
    perm = torch.argsort(-fs, stable=True)
    return fb[perm], fs[perm], fl[perm]
