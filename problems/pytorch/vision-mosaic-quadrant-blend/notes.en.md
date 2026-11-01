# YOLOv4 Mosaic Quadrant-Blend Truth Ledger

Mosaic pastes four images into one net-sized canvas along the cut point
(cut_x, cut_y), each contributing the complementary quadrant. The official
implementation has two layers: the image side is a four-branch memcpy in
`load_data_detection` (copying the current image's sub-block into the target
quadrant), and the truth side is `blend_truth_mosaic` — quadrant-shift each
box, truncate pixel corners, correct out-of-bounds boxes, then apply the
final gate. This problem transcribes both chains from AlexeyAB/darknet
`src/data.c`; all random quantities (cut/shift generation) are explicit
inputs, only the coordinate ledger is tested, no images are read.

## Interface

```python
solve(w, h, cut_x, cut_y, i_mixup, shifts, old_truths, mosaic_bound=False)
```

- `w, h`: net input size (also feeds the lowest gate: `lowest_w = 1/w`).
- `i_mixup ∈ {0,1,2,3}`: the current image's quadrant (0 TL / 1 TR / 2 BL /
  3 BR).
- `shifts = [left_shift, right_shift, top_shift, bot_shift]`: the official
  `min(cut, max(0, −pleft·w/ow))` crop compensation, explicit input.
- `old_truths`: the image's normalized `[x, y, w, h, id]` after
  fill_truth_detection.

## 1. Quadrant shifts (asymmetric four branches)

```
0: xb −= (w−cut_x−right_shift)/w ; yb −= (h−cut_y−bot_shift)/h
1: xb += (cut_x−left_shift)/w    ; yb −= (h−cut_y−bot_shift)/h
2: xb −= (w−cut_x−right_shift)/w ; yb += (cut_y−top_shift)/h
3: xb += (cut_x−left_shift)/w    ; yb += (cut_y−top_shift)/h
```

wb/hb invariant under the shift.

## 2. Pixel corners

`left=(int)((xb−wb/2)·w)` etc. (C int truncation).

## 3. Out-of-bounds correction (mutually exclusive branches)

- **mosaic_bound=True**: clamp to quadrant bounds (0:(0,cut_x,0,cut_y),
  1:(cut_x,w,0,cut_y), 2:(0,cut_x,cut_y,h), 3:(cut_x,w,cut_y,h)) — left<lb
  pulls to lb, right>rb pulls to rb (one-sided), then **recompute** from the
  corners `xb=(right+left)/2/w, wb=(right−left)/w` etc.
- **mosaic_bound=False**: four-direction shrink corrections, **executed
  BEFORE the final gate and able to rescue out-of-bounds boxes**:
  ```
  left<0:  diff=left/w (negative); xb −= diff/2; wb += diff   # shrink
  right>w: diff=(right−w)/w;  xb −= diff/2; wb −= diff
  top<0 / bot>h analogously on yb/hb
  ```
  After all corrections the corners are recomputed (truncated again). A box
  centered exactly at (0,0)/(1,0) is lifted back into bounds rather than
  dropped — the easiest point to miscalculate by hand.

## 4. Final gate (single AND, 14 conditions)

`left≥0 && right≤w && top≥0 && bot≤h && 0<wb<1 && 0<hb<1 && 0<xb<1 &&
0<yb<1 && wb>1/w && hb>1/h` — only then write `[xb, yb, wb, hb, id]`
(id pass-through); otherwise the box is dropped (index recorded only).

## 5. copy_plan (derived from the memcpy branches, current quadrant)

```
0: dst (0,0)         src (w−cut_x−right_shift, h−cut_y−bot_shift)  size (cut_x, cut_y)
1: dst (cut_x,0)     src (left_shift,          h−cut_y−bot_shift)  size (w−cut_x, cut_y)
2: dst (0,cut_y)     src (w−cut_x−right_shift, top_shift)          size (cut_x, h−cut_y)
3: dst (cut_x,cut_y) src (left_shift,          top_shift)          size (w−cut_x, h−cut_y)
```

(Image 0 contributes its bottom-right sub-block, 1 its bottom-left, 2 its
top-right, 3 its top-left.)

## Output

`{"boxes": [[x,y,w,h,id]...], "dropped_indices": [...], "copy_plan": {...}}`.

## Judging details

- A wb==1 (full-width) box is blocked by the `wb<1` gate — the catch for the
  `<=` mutation (truth x=0, wb=1, quadrant 1 → xb=0.5 full box fully in
  bounds).
- The oracle merges dx/dy first then computes per box; 250 random cases
  (shifts aligned to the quadrant, mosaic_bound 50/50).

## Boundaries

- `vision-yolov1-grid-responsibility`: v1's constrain is "clamp then
  recompute"; this problem's bound branch shares that spirit but clamps to
  quadrant bounds; the non-bound shrink corrections and the half-open final
  gate are mosaic-specific.
- mixup=1 (two-image 0.5 blend, blend_truth) and cutmix out of scope; image
  pixels are not read, only the copy_plan coordinates are tested.

## Mutation record

13/13 caught: quadrant shift signs, wrong shift side, bound clamp direction,
no recompute after bound, shrink→expand, top-correction sign, gate `wb<=1`
(caught by the wb==1 anchor), halved lowest, rounded pixel corners,
(r−l)/2 center recompute, id not passed through, copy_plan wrong shift,
unnormalized quadrant shift.
