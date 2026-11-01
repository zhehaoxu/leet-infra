# YOLOv3 Anchor-Box Encode/Decode Ledger

Since YOLOv2, anchor priors are used: the grid only regresses corrections
relative to anchors. This problem transcribes the full ledger of
pjreddie/darknet `f86901f6` (master final) `src/yolo_layer.c` —
`get_yolo_box` (decode), `delta_yolo_box` (encode targets), and the inline
best-anchor matching in forward — nailing down the official **asymmetry**
between encode and decode.

## Interface

```python
solve(lw, lh, img_w, img_h, anchors, truths, preds)
```

- `anchors`: pixel-unit `[pw, ph]` lists (darknet biases).
- `truths`: normalized `[x, y, w, h]` (center format).
- `preds`: aligned with truths; each `[tx, ty, tw, th]` is the **pre-activation
  raw output** at the truth's cell and matched anchor (officially tx/ty get
  LOGISTIC in forward and tw/th stay linear, so raw is given here and the
  activation happens inside decode).

## 1. Best-anchor matching (inline in forward)

```c
i = (truth.x * l.w); j = (truth.y * l.h);
truth_shift.x = truth_shift.y = 0;      // shift center to origin
pred.w = l.biases[2*n]/net.w; ...       // normalized anchor
iou > best_iou to update                 // strict >: ties to lower index
```

wh-only, center-aligned: `inter=min(w1,w2)·min(h1,h2)`; `best_iou` starts at
0. (Changing `best_iou` to −1 is an **equivalent mutation**: IoU≥0 always
holds, so both the selection and the output are identical.)

## 2. Cell assignment

`i=(int)(truth.x·lw), j=(int)(truth.y·lh)` — int truncation, no official clamp.

## 3. Encode targets (delta_yolo_box)

```
tgt_x = truth.x·lw − i              # linear offset, NOT logit(σ⁻¹)!
tgt_w = log(truth.w·img_w / pw)     # log ratio
```

**Core asymmetry**: decode applies `σ(tx)` while the encode target is a
linear difference — the xy encode/decode round trip does **not** close (the
official loss is L2 on σ(tx) with the target in σ-output space); wh's
log/exp is strictly invertible (`e^tgt_w·pw/img_w == truth.w`).

## 4. Decode (get_yolo_box)

```
bx = (i + σ(tx)) / lw
by = (j + σ(ty)) / lh
bw = e^tw · pw / img_w              # pixel biases / image size
bh = e^th · ph / img_h
```

Channel layout: `stride = lw·lh`, channel-separated (outputs here are logical
values; stride is not expanded).

## 5. Delta

`delta = scale · (target − output)` with `scale = 2 − truth.w·truth.h`
(normalized truth); the xy output is `σ(tx)`, the wh output is raw.

## Judging details

- Anchor ties: strict `>`, first maximum wins (lower index).
- The `basic_single` anchor: raw all-zero → σ=0.5 → `bx=(i+0.5)/lw`; the wh
  delta equals `scale·tgt_w` (raw=0).
- The oracle is a functional direct computation (`max(range, key=(iou, -n))`
  organization), cross-checked over 250 random cases (lw/lh∈{4,7,13},
  1–5 anchors, 1–4 truths).

## Boundaries

- `vision-yolov1-grid-responsibility`: anchor-free grid system (sqrt
  encoding, linear offsets, first-come-first-served); this problem is the
  anchor system (exp/log encoding, IoU matching, sigmoid offsets).
- `vision-yolo-anchor-kmeans`: the clustering source of the anchor priors.
- Masks (multi-scale anchor subsets), objectness/classes channels, and NMS
  post-processing are out of scope.

## Mutation record

12 non-equivalent mutations all caught: ties to last-writer, matching as
1−|log ratio| (k-means confusion), decode without σ, bw over grid width,
tgt_x+0.5, tgt_w parenthesization, scale 1−wh, delta addition, cell round,
denominator swap, wh encode without anchor division, unconditional last
anchor. One **equivalent mutation** (`best_iou=−1` init) recorded in notes.
