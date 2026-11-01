# YOLOv3 Anchor-Box Encode/Decode Ledger

## Problem

YOLOv2/YOLOv3 use anchor priors: a predicted box = anchor + network
correction. In the official darknet implementation, encode (training targets)
and decode (inference restoration) are **asymmetric** — the xy target is a
linear offset while decode applies sigmoid, and only wh is strictly
log/exp-invertible. Given the grid, anchor priors, and a set of truths and
raw network outputs, reproduce the official ledger line by line.

Given:
- `lw, lh`: grid columns/rows (ints ≥1)
- `img_w, img_h`: image size (the normalization base)
- `anchors`: pixel-unit `[pw, ph]` lists
- `truths`: normalized `[x, y, w, h]` (center format)
- `preds`: aligned with truths; each `[tx, ty, tw, th]` is the raw
  (pre-activation) output at the cell of the matched anchor

## Processing chain (per truth, in order)

1. **Best-anchor matching**: shift the truth center to the origin, compute
   the wh-only center-aligned IoU against each anchor's normalized
   `(pw/img_w, ph/img_h)` (`inter=min(w1,w2)·min(h1,h2)`,
   `union=w1·h1+w2·h2−inter`); update only on `iou > best_iou` (strict,
   initial 0) → ties to the lower index.
2. **Cell assignment**: `i=(int)(truth.x·lw), j=(int)(truth.y·lh)`
   (int truncation).
3. **Encode targets**: `tgt_x=truth.x·lw−i`, `tgt_y=truth.y·lh−j` (linear
   offsets); `tgt_w=log(truth.w·img_w/pw)`, `tgt_h=log(truth.h·img_h/ph)`
   (log ratios, pw/ph of the matched anchor).
4. **Decode**: `bx=(i+σ(tx))/lw`, `by=(j+σ(ty))/lh`, `bw=e^tw·pw/img_w`,
   `bh=e^th·ph/img_h`.
5. **Delta**: `scale=2−truth.w·truth.h` (normalized truth);
   `delta=[scale·(tgt_x−σ(tx)), scale·(tgt_y−σ(ty)), scale·(tgt_w−tw),
   scale·(tgt_h−th)]`.

## Output

`{"matches": [{...}, ...]}` in truth order; each item contains
`truth_index`, `anchor` (matched anchor index), `cell: [i, j]`,
`target: [tgt_x, tgt_y, tgt_w, tgt_h]`, `pred_box: [bx, by, bw, bh]`,
`delta: [4 values]`, and `anchor_iou` (the matched IoU).

## Input constraints

- `lw/lh` are ints ≥1; `img_w/img_h > 0`; `anchors` non-empty with positive
  pairs; `truths/preds` equal length; truths finite with `w,h>0`; preds are
  4 finite floats; violations raise `ValueError`.

## Example

`lw=lh=7, img=416, anchors=[[32,64]], truth=[0.55,0.6,0.2,0.3],
pred=[0,0,0,0]`: cell `[3,4]`; `tgt_x=0.85, tgt_w=log(0.2·416/32)`;
`bx=(3+0.5)/7=0.5, bw=32/416`; `scale=2−0.06=1.94`;
`delta[0]=1.94·(0.85−0.5)`.
