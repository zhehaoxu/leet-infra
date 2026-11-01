# YOLOv1 Grid Responsibility Encoding Ledger

## Problem

YOLOv1 casts detection as regression: the image is split into an S×S grid,
and the grid cell containing a GT center takes responsibility for that
target. In the official C implementation, however, "encoding" is a chain full
of pitfalls — after jitter the corners are clamped, and after clamping the
box size must be **recomputed**; the center gate is half-open; cell
assignment uses int truncation with no clamp; the 0.01 size gate is strict
less-than; and same-cell conflicts are first-come-first-served.

Given `side` (S), the number of classes `classes`, a label list
(`[id, x, y, w, h]`, center and size in normalized image coordinates, in the
post-shuffle order), jitter parameters `[dx, dy, sx, sy]`, and a `flip` flag,
reproduce the official encode chain line by line, and produce the decode
result of every occupied cell under the `convert_detections` convention.

## Processing chain (per label, in order)

1. Corner expansion: `left=x−w/2, right=x+w/2, top=y−h/2, bot=y+h/2`.
2. Jitter: `left=left·sx−dx` etc. (same form for right/bottom).
3. Flip (if on): `left, right = 1−right, 1−left`.
4. `constrain(0,1)`: clamp each corner into [0,1].
5. **Recompute** center and size: `x=(left+right)/2, y=(top+bot)/2,
   w=right−left, h=bot−top` — the size is the post-clamp corner difference,
   not a clamp of the original w.
6. Center gate: `x<=0 or x>=1 or y<=0 or y>=1` → drop (half-open interval;
   centers exactly on the boundary are dropped too), recording
   `{"index": i, "reason": "center_out"}`.
7. Cell assignment: `col=(int)(x·S), row=(int)(y·S)` (int truncation; thanks
   to step 6, no clamp needed). In-cell offsets: `tx=x·S−col, ty=y·S−row`
   (multiply first, subtract second).
8. Size gate: post-clamp `w<0.01 or h<0.01` → drop (strict `<`; 0.01
   survives), recording `reason: "too_small"`.
9. sqrt encoding: `tw=w^0.5, th=h^0.5`.
10. Occupancy arbitration: if cell `(row, col)` is taken → drop, first come
    first served, recording `reason: "occupied"`.
11. Write cell: `grid[f"{row},{col}"] = {"id", "x": tx, "y": ty, "w": tw,
    "h": th, "conf": 1.0}` (the confidence target is always 1).

## Decode (inverse)

For every occupied cell: `cx=(col+tx)/S, cy=(row+ty)/S, bw=tw², bh=th²`,
stored in `decoded` under the same key as a list `[cx, cy, bw, bh]`.

## Input constraints

- `side`, `classes` are ints ≥1; `jitter` is 4 finite floats (default
  `[0,0,1,1]`); `flip` is bool; each label has `id ∈ [0,classes)` and
  `w,h > 0`; violations raise `ValueError`.

## Example

`side=7, classes=20, labels=[[3, 0.55, 0.6, 0.2, 0.4]]`: center (0.55,0.6)
→ col=3, row=4; `tx=0.55·7−3≈0.85, ty≈0.2`; `tw=√0.2, th=√0.4`; output cell
`"4,3"`; decode returns `[0.55, 0.6, 0.2, 0.4]`.
