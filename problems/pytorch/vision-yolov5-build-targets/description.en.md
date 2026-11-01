# YOLOv5 build_targets Double-Expansion Assignment Ledger

## Problem

YOLOv5's positive-sample assignment is much wider than "center falls in
cell": each GT is first coarse-filtered by anchor shape, then **expanded
toward neighbor cells** on the fractional side (offset 0.5) — one GT can
land on up to 5 grid cells at once. Given the GT list, per-layer anchors
and feature-map sizes, reproduce the official `build_targets` ledger
(single-image form).

Given:
- `targets`: nt x 5 `[class, x, y, w, h]` (normalized; x,y∈[0,1],
  w,h∈(0,1])
- `anchors`: nl layers x na anchors x `[w, h]` (pixel units)
- `shapes`: nl layers `[H, W]` (grid sizes)
- `anchor_t`: the wh-ratio threshold (default 4.0)

## Processing chain (per layer, independent)

1. **Replicate + gain**: each GT replicated na times with its anchor
   index; gain = `[1, 1, W, H, W, H, 1]` scales into grid space
   (x with W, y with H).
2. **wh match**: `r = grid-space wh / pixel anchor` (official mixup),
   `max(r, 1/r) < anchor_t` after per-axis max (**equality filters too**).
   ⚠️ For the same target r shrinks 4x per layer (grid width ÷2, anchor
   ×2) — eligibility swings wildly across layers.
3. **Five-direction expansion**: offsets = `[0,0], [+0.5,0], [0,+0.5],
   [−0.5,0], [0,−0.5]`. A direction fires (x/y independent) when:
   - same side (j/k): `coord % 1 < 0.5` **AND** `coord > 1`;
   - opposite side (l/m): `(W−x) % 1 < 0.5` **AND** `(W−x) > 1`
     (y uses H).
   Near-edge coordinates (coord or inverse ≤1) never fire; fraction
   exactly 0.5 never fires.
4. **Grid cell**: `gij = (gxy − offset).long()` — **truncation, not
   round**. When x is an exact integer, dir0 and dir3 land on the same
   cell.
5. **clamp**: `gj∈[0,H−1], gi∈[0,W−1]` (official defensive branch,
   unreachable under the dual gate).
6. **Output**: indices = `[a, gj, gi]`; tbox = `[gx−gij_x, gy−gij_y,
   gw, gh]` (residual against the truncated cell); anch = the anchor's
   `[w, h]`.

## Output ledger (one entry per layer)

```json
{
  "gain":     [1, 1, W, H, W, H, 1],
  "wh_keep":  [bool × (na·nt)],                    // anchor-major
  "expanded": [[a, cls, gx, gy, gw, gh, dir], …],  // dir-major order
  "indices":  [[a, gj, gi], …],
  "tbox":     [[dx, dy, gw, gh], …],
  "anch":     [[aw, ah], …]
}
```

## Input constraints

- targets rows: 5 finite columns; x,y∈[0,1], w,h∈(0,1]; class a
  non-negative int;
- anchors: 2 positive finite values each; shapes: 2 ints ≥1;
  anchor_t>0; violations raise `ValueError`; empty targets is legal
  (all layers empty).

## Example

`anchors=[[[8,8]]], shapes=[[16,16]]`, target `[0, 14.6/16, 8.5/16,
0.2, 0.4]`: grid gx=14.6, gy=8.5, gw=3.2, gh=3.2 (r=0.4, keep).
gx%1=0.6 → j False; gxi=1.4 (%1=0.4 and >1) → l True; y side both False
→ dirs=[0,3]. dir0: gij=int(14.6)=14, residual 0.6; dir3:
int(14.6+0.5)=15, residual 0.1 → indices `[[0,4,14],[0,4,15]]`.
