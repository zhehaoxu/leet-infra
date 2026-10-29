import torch


def solve(gt_boxes, points, levels, strides, regress_ranges, center_radius=1.5):
    if not isinstance(gt_boxes, torch.Tensor) or gt_boxes.dtype not in (torch.float32, torch.float64):
        raise ValueError("gt_boxes 必须是 float32/float64 张量")
    if gt_boxes.dim() != 2 or gt_boxes.shape[1] != 4:
        raise ValueError(f"gt_boxes 形状必须是 (G,4)，got {tuple(gt_boxes.shape)}")
    if not isinstance(points, torch.Tensor) or points.dtype != gt_boxes.dtype:
        raise ValueError("points 必须是张量且与 gt_boxes 同 dtype")
    if points.dim() != 2 or points.shape[1] != 2:
        raise ValueError(f"points 形状必须是 (P,2)，got {tuple(points.shape)}")
    if len(strides) != len(regress_ranges) or len(strides) == 0:
        raise ValueError("strides 与 regress_ranges 必须等长且非空")
    if any(float(s) <= 0 for s in strides):
        raise ValueError("strides 必须为正")
    if any(float(lo) >= float(hi) for lo, hi in regress_ranges):
        raise ValueError("regress_ranges 每项必须满足 lo < hi")
    if float(center_radius) < 0:
        raise ValueError("center_radius 必须非负")

    lv = torch.as_tensor(levels, dtype=torch.int64).reshape(-1)
    P = points.shape[0]
    if lv.numel() != P:
        raise ValueError(f"levels 长度必须等于点数 {P}，got {lv.numel()}")
    num_levels = len(strides)
    if P > 0 and (int(lv.min()) < 0 or int(lv.max()) >= num_levels):
        raise ValueError(f"levels 取值必须在 [0,{num_levels}) 内")

    dtype = gt_boxes.dtype
    labels = torch.full((P,), -1, dtype=torch.int64)
    centerness = torch.zeros(P, dtype=dtype)
    reg_targets = torch.zeros(P, 4, dtype=dtype)
    if gt_boxes.shape[0] == 0 or P == 0:
        return labels, centerness, reg_targets

    stride_pt = torch.as_tensor([float(strides[int(i)]) for i in lv], dtype=dtype)
    lo_pt = torch.as_tensor([float(regress_ranges[int(i)][0]) for i in lv], dtype=dtype)
    hi_pt = torch.as_tensor([float(regress_ranges[int(i)][1]) for i in lv], dtype=dtype)

    x1, y1, x2, y2 = (gt_boxes[:, i] for i in range(4))
    px = points[:, 0:1]
    py = points[:, 1:2]
    l = px - x1[None, :]
    t = py - y1[None, :]
    r = x2[None, :] - px
    b = y2[None, :] - py
    inside = (l >= 0) & (t >= 0) & (r >= 0) & (b >= 0)

    radius = (center_radius * stride_pt).unsqueeze(1)
    cx = ((x1 + x2) * 0.5)[None, :]
    cy = ((y1 + y2) * 0.5)[None, :]
    in_center = ((px - cx).abs() <= radius) & ((py - cy).abs() <= radius)

    maxd = torch.stack([l, t, r, b], dim=-1).max(dim=-1).values
    in_range = (maxd >= lo_pt.unsqueeze(1)) & (maxd < hi_pt.unsqueeze(1))

    cand = inside & in_center & in_range
    area = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0))[None, :]
    pick = torch.where(cand, -area, torch.full_like(area, torch.finfo(dtype).min))
    _, best_gt = pick.max(dim=1)
    has = cand.any(dim=1)

    covered = inside.any(dim=1)
    labels = torch.where(has, best_gt, torch.where(covered, labels.new_full((P,), -2), labels))

    idx = has.nonzero(as_tuple=False).flatten()
    g = best_gt[idx]
    ls, ts, rs, bs = l[idx, g], t[idx, g], r[idx, g], b[idx, g]
    reg_targets[idx] = torch.stack([ls, ts, rs, bs], dim=-1)
    lr_min, lr_max = torch.minimum(ls, rs), torch.maximum(ls, rs)
    tb_min, tb_max = torch.minimum(ts, bs), torch.maximum(ts, bs)
    zero = torch.zeros_like(lr_max)
    f_lr = torch.where(lr_max > 0, lr_min / lr_max, zero)
    f_tb = torch.where(tb_max > 0, tb_min / tb_max, zero)
    centerness[idx] = (f_lr * f_tb).sqrt()
    labels[idx] = g
    return labels, centerness, reg_targets
