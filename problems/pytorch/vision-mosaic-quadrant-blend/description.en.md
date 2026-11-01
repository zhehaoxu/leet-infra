# YOLOv4 Mosaic Quadrant-Blend Truth Ledger

## Problem

Mosaic augmentation pastes four images' complementary quadrants into one
training image. The truth side involves more than "shifting" — quadrant
directions are asymmetric, pixel corners get int-truncated, out-of-bounds
boxes have two correction branches (bound: clamp to quadrant bounds and
recompute; non-bound: shrink corrections executed **before the final gate**,
able to rescue boxes), and finally a 14-condition AND gate. Given the paste
parameters and one image's box list, reproduce the official
`blend_truth_mosaic` ledger.

Given:
- `w, h`: net input size (ints ≥1)
- `cut_x, cut_y`: cut point (0≤cut_x≤w, 0≤cut_y≤h)
- `i_mixup`: the current image's quadrant 0..3 (TL/TR/BL/BR)
- `shifts = [left_shift, right_shift, top_shift, bot_shift]`: non-negative
  ints
- `old_truths`: normalized `[x, y, w, h, id]` lists
- `mosaic_bound`: bool

## Processing chain (per box, in order)

1. Quadrant shift (asymmetric four branches):
   `0: xb−=(w−cut_x−right_shift)/w, yb−=(h−cut_y−bot_shift)/h`;
   `1: xb+=(cut_x−left_shift)/w, yb−=(h−cut_y−bot_shift)/h`;
   `2: xb−=(w−cut_x−right_shift)/w, yb+=(cut_y−top_shift)/h`;
   `3: xb+=(cut_x−left_shift)/w, yb+=(cut_y−top_shift)/h`.
2. Pixel corners: `left=(int)((xb−wb/2)·w)` etc.
3. Out-of-bounds correction (mutually exclusive branches):
   - mosaic_bound=True: one-sided clamp to the quadrant bounds
     (left<lb→lb, right>rb→rb, top/bot likewise), then recompute from
     corners `xb=(right+left)/2/w`, `wb=(right−left)/w` etc.;
   - mosaic_bound=False: `left<0: diff=left/w, xb−=diff/2, wb+=diff`
     (diff<0, i.e. shrink); `right>w: diff=(right−w)/w, xb−=diff/2, wb−=diff`;
     top/bot likewise; corners recomputed after all corrections.
4. Final gate (single AND): `left≥0 && right≤w && top≥0 && bot≤h &&
   0<wb<1 && 0<hb<1 && 0<xb<1 && 0<yb<1 && wb>1/w && hb>1/h`.
5. Pass → `boxes.append([xb, yb, wb, hb, id])` (id pass-through); otherwise
   record the index.

## copy_plan (memcpy-derived, current quadrant)

```
0: dst (0,0)         src (w−cut_x−right_shift, h−cut_y−bot_shift)  size (cut_x, cut_y)
1: dst (cut_x,0)     src (left_shift,          h−cut_y−bot_shift)  size (w−cut_x, cut_y)
2: dst (0,cut_y)     src (w−cut_x−right_shift, top_shift)          size (cut_x, h−cut_y)
3: dst (cut_x,cut_y) src (left_shift,          top_shift)          size (w−cut_x, h−cut_y)
```

## Output

`{"boxes": [...], "dropped_indices": [...], "copy_plan": {dst, src, size}}`.

## Input constraints

- w/h ints ≥1; cuts in range; i_mixup∈0..3; shifts are 4 non-negative ints;
  truths `[x,y,w,h,id]` finite with w,h>0; mosaic_bound bool; violations
  raise `ValueError`.

## Example

`w=h=416, cut=(208,208), i_mixup=1, shifts=0`, truth
`[0.3,0.6,0.1,0.1,7]` → shift (xb,yb)=(0.8,0.1), passes the gate unchanged,
output box `[0.8, 0.1, 0.1, 0.1, 7]`; copy_plan
`dst=[208,0], src=[0,208], size=[208,208]`.
