"""YOLOv5 build_targets double-expansion assignment, official path.

Reproduced from ultralytics/yolov5 v7.0 `utils/loss.py::build_targets`
(single-image ledger; the batch/image column is omitted).

Per detection layer:
  1. gain = [1, 1, W, H, W, H, 1]; each GT is replicated per anchor with
     its anchor index appended, then scaled into grid space
  2. wh match: max(r, 1/r).max() < anchor_t per axis where
     r = grid-space wh / pixel anchor wh (official pixel/grid mixup)
  3. offsets expansion: five directions [0,0],[1,0],[0,1],[-1,0],[0,-1]
     scaled by g=0.5; direction d fires when the coordinate fraction
     < 0.5 on the expanding side AND the coordinate (or its inverse
     W-x) > 1 (edge guard)
  4. grid cell: gij = (gxy - offsets).long()  (truncation, not round)
     then clamp to [0, shape-1] (defensive; the >1 guards already prevent
     out-of-bounds in normal grids)
  5. tbox = (gxy - gij, gwh); anch = the anchor wh of the assigned cell

Returns a per-layer ledger with the wh-match mask, expanded rows (with
the firing direction), final indices, tbox and anchor whs.
"""
import math


def solve(targets, anchors, shapes, anchor_t=4.0):
    nl = len(anchors)
    if nl == 0:
        raise ValueError("need at least one layer")
    na = len(anchors[0])
    if na == 0:
        raise ValueError("need at least one anchor")
    if len(shapes) != nl:
        raise ValueError("shapes must match layers")
    for layer in anchors:
        for a in layer:
            if len(a) != 2 or a[0] <= 0 or a[1] <= 0 \
                    or not all(math.isfinite(v) for v in a):
                raise ValueError("anchor wh must be positive finite")
    for s in shapes:
        if len(s) != 2 or s[0] < 1 or s[1] < 1:
            raise ValueError("shape must be (H, W) with dims >= 1")
    if not (anchor_t > 0):
        raise ValueError("anchor_t must be positive")
    for t in targets:
        if len(t) != 5:
            raise ValueError("target rows must be [class, x, y, w, h]")
        if not all(math.isfinite(v) for v in t):
            raise ValueError("target values must be finite")
        if t[1] < 0 or t[1] > 1 or t[2] < 0 or t[2] > 1 \
                or t[3] <= 0 or t[3] > 1 or t[4] <= 0 or t[4] > 1:
            raise ValueError("x,y in [0,1], w,h in (0,1]")
        if t[0] != int(t[0]) or t[0] < 0:
            raise ValueError("class must be a non-negative int")

    g = 0.5
    offs = [(0.0, 0.0), (g, 0.0), (0.0, g), (-g, 0.0), (0.0, -g)]

    layers = []
    for i in range(nl):
        W, H = float(shapes[i][1]), float(shapes[i][0])
        gain = [1.0, 1.0, W, H, W, H, 1.0]

        # replicate per anchor: [a, cls, gx, gy, gw, gh] in grid space
        t3 = []
        for a in range(na):
            for t in targets:
                t3.append([float(a), t[0], t[1] * gain[2], t[2] * gain[3],
                           t[3] * gain[4], t[4] * gain[5]])

        # wh match
        keep = []
        for row in t3:
            ok = True
            for dim in range(2):
                r = row[4 + dim] / anchors[i][int(row[0])][dim]
                if max(r, 1.0 / r) >= anchor_t:
                    ok = False
                    break
            keep.append(ok)
        t2 = [row for row, ok in zip(t3, keep) if ok]

        # offsets expansion
        expanded = []
        for row in t2:
            gx, gy = row[2], row[3]
            gxi, gyi = W - gx, H - gy
            dirs = [True,
                    (gx % 1.0 < g) and (gx > 1.0),
                    (gy % 1.0 < g) and (gy > 1.0),
                    (gxi % 1.0 < g) and (gxi > 1.0),
                    (gyi % 1.0 < g) and (gyi > 1.0)]
            for d in range(5):
                if not dirs[d]:
                    continue
                ox, oy = offs[d]
                expanded.append((row, d, ox, oy))

        indices, tbox, anch_rows = [], [], []
        for row, d, ox, oy in expanded:
            gx, gy = row[2] - ox, row[3] - oy
            gij_x = int(gx)      # official .long(): truncation
            gij_y = int(gy)
            gj = min(max(gij_y, 0), int(H) - 1)
            gi = min(max(gij_x, 0), int(W) - 1)
            indices.append([int(row[0]), gj, gi])
            tbox.append([row[2] - gij_x, row[3] - gij_y,
                         row[4], row[5]])
            anch_rows.append(list(anchors[i][int(row[0])]))
        layers.append({
            "gain": gain,
            "wh_keep": keep,
            "expanded": [[int(r[0]), r[1], r[2], r[3], r[4], r[5], d]
                         for r, d, _, _ in expanded],
            "indices": indices,
            "tbox": tbox,
            "anch": anch_rows,
        })
    return layers
