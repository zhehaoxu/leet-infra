# YOLOv1 Grid Responsibility Encoding Ledger

YOLOv1 uses no anchors: the image is split into an S×S grid, the grid cell
containing a GT center takes responsibility for that target, coordinates are
encoded as in-cell offsets, and wh goes through sqrt compression. This problem
transcribes the full official ledger — not the idealized paper formulas, but
the line-level conventions of `src/data.c` `fill_truth_detection` and
`src/detection.c` `convert_detections` at pjreddie/darknet commit `9db61832`
(2015-07-20, around the CVPR camera-ready).

## Interface

```python
solve(side, classes, labels, jitter=None, flip=False)
```

- `labels`: `[id, x, y, w, h]` lists where `x,y` are **normalized centers**
  and `w,h` normalized sizes (`read_boxes` convention). List order equals the
  post-`randomize_boxes` shuffled order — randomness is an explicit input.
- `jitter = [dx, dy, sx, sy]`: the official training translate/scale jitter,
  applied to corners.
- Output `{"grid": {"r,c": {...}}, "decoded": {...}, "dropped": [...]}`.

## Encode side (fill_truth_detection)

1. **Corner expansion**: `left=x−w/2, right=x+w/2, top=y−h/2, bot=y+h/2`.
2. **Jitter**: `left=left·sx−dx` (same form for right/bottom).
3. **Flip**: `left, right = 1−right, 1−left` (new left comes from old right).
4. **constrain(0,1) per corner**.
5. **Recompute**: `x=(left+right)/2, y=(top+bot)/2, w=right−left, h=bot−top`.
   Key ledger point: `w,h` are the **post-clamp corner differences** — when a
   corner is clamped the size changes; never `clamp(w, 0, 1)`.
6. **Half-open center gate**: `x<=0 or x>=1 or y<=0 or y>=1 → continue`.
   A center exactly at 0 or 1 is dropped, not clamped.
7. **Cell assignment**: `col=(int)(x·S), row=(int)(y·S)` — C int truncation.
   Step 6 already guarantees `x·S ∈ (0,S)`, so the official code has **no
   clamp**; a center exactly on a grid line (x·S=1.0) legally lands in the
   next cell with offset 0.
8. **In-cell offset**: `tx=x·S−col, ty=y·S−row` — multiply first, subtract
   second; floating-point sensitive (not equal to `x−col/S`).
9. **Size gate**: post-clamp `w<0.01 or h<0.01 → continue` (strict `<`; 0.01
   itself survives).
10. **sqrt encoding**: `tw=w^0.5, th=h^0.5` (official `pow(w,1./2.)`; the
    commented-out `sqrt(w)` is equivalent).
11. **Occupancy arbitration**: `if(truth[index+classes+2]) continue;` — if the
    cell already holds an object, later GTs are dropped, **first come first
    served**, in shuffled order.

Truth slot layout `(4+classes)`: one-hot classes + `[tx, ty, tw, th]`.
**Confidence target is always 1** — the truth array has no conf slot; v1
confidence supervision lives in the loss/objectness branch. The official
background flag slot and rescore (conf←IoU) conventions are out of scope.

## Decode side (convert_detections)

Row-major expansion: `row=i//S, col=i%S`, then

```
cx = (px + col) / S      # px = network output = tx
cy = (py + row) / S
bw = px^2                # square restores the sqrt encoding
bh = py^2
```

inverse of encode (this problem outputs normalized coords, no pixel w/h).

## Judging details

- Exact float boundary: `0.505−0.495 = 0.010000000000000009 > 0.01`, so the
  "w exactly 0.01 survives" anchor needs a dedicated construction —
  `l=0.0, r=0.01` (e.g. `x=0.005, w=0.01`) is the exact float hit.
- The oracle is a mathematically equivalent direct-form implementation: the
  decode branch writes the post-clamp `x,y,w,h` directly (round-trip squaring
  differs by ~1e-17, within tol 1e-9), cross-checked against the solution's
  C-chain replay over 300 random cases (S∈{1,2,3,5,7}, with jitter/flip).

## Boundaries

- `vision-anchor-match`: IoU-threshold anchor matching with anchor priors;
  this problem is the anchor-free grid system.
- `vision-yolo-anchor-kmeans`: the clustering source of anchor priors; this
  problem is grid responsibility.
- Background flag slot, rescore, and loss-side IOU restoration are out of
  scope.

## Official loss-side quirk (recorded, not in the problem)

In `detection_layer.c` forward, `truth.x = truth[j+0]/7` uses the in-cell
offset divided by S directly as the image coordinate, **without adding col**
(S hardcoded to 7). It is a crude IOU/diou estimate whose x/y gradient is
immediately overwritten by `4*(truth−output)` — effectively dead code; the
detection-output side `convert_detections` is the correct `(px+col)/S`.

## Mutation record (12/12 caught)

post-clamp w not recomputed, center gate made closed-open, col rounded,
offset +0.5, size gate `<=` (caught by the exact-float boundary anchor),
encode squared, last-writer-wins occupancy, flip disabled, decode row/col
swapped, decode unsquared, int→round, conf 0.5.
