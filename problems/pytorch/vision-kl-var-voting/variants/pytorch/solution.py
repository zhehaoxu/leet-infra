"""KL-Loss (CVPR'19) variance voting + soft-NMS, official stdiou branch.

Reproduced from ethanhe42/KL-Loss `detectron/utils/py_cpu_nms.py::soft`
with cfg.STD.METHOD == 'stdiou' (the variance-voting path of
"Bounding Box Regression with Uncertainty for Accurate Object Detection").

Input:
  dets      : N x 5  [x1, y1, x2, y2, score]  (float)
  confs     : N x 4  per-coordinate sigma (x1, y1, x2, y2) predicted by
              the KL-Loss head; smaller = more certain
  thresh    : STD_TH; if > 0.1 it is clamped to 0.01 (official quirk)
  iou_sigma : cfg.STD.IOU_SIGMA, kernel width of the Gaussian vote weight

Pipeline (all in one pass, official order):
  1. pairwise IoU with the legacy VOC "+1" pixel convention
  2. greedy loop: pick argmax score among remaining, swap it to slot i
     (swapping dets, confs, orig index and IoU rows/cols together)
  3. voters = remaining boxes with IoU > thresh (strictly greater)
  4. fused coord_j = sum_k p_k * coord_kj / s_kj^2
                    / sum_k p_k / s_kj^2
     with p_k = exp(-(1 - iou_k)^2 / iou_sigma), per-coordinate weights
  5. suppress: for every later box with iou > 0 (ANY positive overlap),
     score *= exp(-iou^2 / 0.5); drop score < 0.001 by tail-swap
  6. fused coordinates are NOT clipped, and IoU matrix is NOT recomputed
     after fusion (later rounds reuse stale IoUs against fused boxes)

Returns a ledger:
  ious     : initial N x N IoU matrix
  steps    : per-round vote records
  final    : kept boxes in processing order (fused coords, decayed scores)
  removed  : original indices of boxes decayed below the floor
"""
import math


def solve(dets, confs, thresh, iou_sigma):
    dets = [[float(v) for v in row] for row in dets]
    confs = [[float(v) for v in row] for row in confs]
    n_total = len(dets)
    if len(confs) != n_total:
        raise ValueError("confs must match dets length")
    for row in dets:
        if len(row) != 5 or not all(math.isfinite(v) for v in row):
            raise ValueError("each det row must be [x1, y1, x2, y2, score]")
    for row in confs:
        if len(row) != 4 or any(v <= 0 or not math.isfinite(v)
                                for v in row):
            raise ValueError("each conf row must be 4 positive sigmas")
    if not (iou_sigma > 0) or not math.isfinite(iou_sigma):
        raise ValueError("iou_sigma must be positive")
    if not math.isfinite(thresh):
        raise ValueError("thresh must be finite")

    # official: stdiou clamps a loose STD_TH down to 0.01
    if thresh > 0.1:
        thresh = 0.01

    def _iou(a, b):
        # legacy VOC convention: +1 on widths/heights of both inter and areas
        iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]) + 1.0)
        ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]) + 1.0)
        inter = iw * ih
        area_a = (a[2] - a[0] + 1.0) * (a[3] - a[1] + 1.0)
        area_b = (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0)
        return inter / (area_a + area_b - inter)

    ious = [[_iou(dets[i], dets[j]) for j in range(n_total)]
            for i in range(n_total)]

    # mutable working set: slots [0, n) alive, tail holds decayed-out boxes
    orig = list(range(n_total))
    n = n_total
    ious_rows = [row[:] for row in ious]

    steps = []
    removed = []
    i = 0
    while i < n:
        # argmax among dets[i:n], swap the winner into slot i
        maxpos = i
        best = dets[i][4]
        for k in range(i + 1, n):
            if dets[k][4] > best:
                best = dets[k][4]
                maxpos = k
        if maxpos != i:
            dets[i], dets[maxpos] = dets[maxpos], dets[i]
            confs[i], confs[maxpos] = confs[maxpos], confs[i]
            orig[i], orig[maxpos] = orig[maxpos], orig[i]
            ious_rows[i], ious_rows[maxpos] = ious_rows[maxpos], ious_rows[i]
            for row in ious_rows:
                row[i], row[maxpos] = row[maxpos], row[i]

        # voters: strictly greater IoU against the current pick
        voters = [k for k in range(i, n) if ious_rows[i][k] > thresh]

        # variance voting: per-coordinate weighted average
        box = [0.0] * 4
        for c in range(4):
            num = 0.0
            den = 0.0
            for k in voters:
                p = math.exp(-(1.0 - ious_rows[i][k]) ** 2 / iou_sigma)
                num += p * dets[k][c] / (confs[k][c] ** 2)
                den += p / (confs[k][c] ** 2)
            box[c] = num / den if den else dets[i][c]
        dets[i][:4] = box

        steps.append({
            "idx": orig[i],
            "box": box[:],
            "voters": sorted(orig[k] for k in voters),
            "weights": [[math.exp(-(1.0 - ious_rows[i][k]) ** 2 / iou_sigma)
                         / (confs[k][c] ** 2) for c in range(4)]
                        for k in sorted(voters, key=lambda k: orig[k])],
        })

        # suppression: ANY later box with positive IoU gets decayed
        pos = i + 1
        while pos < n:
            if ious_rows[i][pos] > 0:
                dets[pos][4] *= math.exp(
                    -(ious_rows[i][pos] ** 2) / 0.5)
                if dets[pos][4] < 0.001:
                    removed.append(orig[pos])
                    dets[pos], dets[n - 1] = dets[n - 1], dets[pos]
                    confs[pos], confs[n - 1] = confs[n - 1], confs[pos]
                    orig[pos], orig[n - 1] = orig[n - 1], orig[pos]
                    ious_rows[pos], ious_rows[n - 1] = \
                        ious_rows[n - 1], ious_rows[pos]
                    for row in ious_rows:
                        row[pos], row[n - 1] = row[n - 1], row[pos]
                    n -= 1
                    pos -= 1
            pos += 1
        i += 1

    return {
        "ious": ious,
        "steps": steps,
        "final": [dets[k] for k in range(n)],
        "removed": removed,
    }
