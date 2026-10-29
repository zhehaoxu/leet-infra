# In Depth: Detection: Rotated-Box NMS and Overlap Area

## Source

- The rotated-box detection paradigm: Xia et al., *DOTA: A Large-scale Dataset for Object Detection in Aerial Images*, CVPR 2018. Aerial targets (ships, planes, storage tanks, courts) are annotated as arbitrarily oriented quadrilaterals or rotated rectangles; DOTA also established the "regress an angle, evaluate with rotated IoU" workflow.
- Polygon clipping: Sutherland & Hodgman, *Reentrant Polygon Clipping*, Communications of the ACM 17(1), 1974. Scan one edge at a time, keep inside vertices and insert intersections — the standard algorithm for clipping a polygon by a convex polygon; clipping two convex quadrilaterals leaves at most \(4+4=8\) vertices.
- Area: the shoelace (Gauss) formula holds for any simple polygon; the intersection here is always convex, so a plain signed sum and absolute value suffice.
- Engineering references: mmcv's `box_iou_rotated` (C++/CUDA polygon clipping) and OpenCV's `cv::rotatedRectangleIntersection`. Neither is a dependency here; the judge only has torch, and the reference implements per-edge clipping in pure torch.
- The strictness of axis-aligned NMS, the greedy order and "ties by smaller index" follow the convention of the repo's own `nms` problem; this problem swaps in rotated IoU and additionally returns the full matrix.
- CPU only, deterministic, no RNG: angles go through `cos/sin`; clipping and shoelace are four arithmetic operations.

## Full derivation and index convention

**Corners.** For a box \( (c_x,c_y,w,h,\theta) \), take the four centre-frame offsets

\[
(+w/2,-h/2),\quad (+w/2,+h/2),\quad (-w/2,+h/2),\quad (-w/2,-h/2),
\]

then rotate by \( \theta \):

\[
p_x = o_x\cos\theta-o_y\sin\theta+c_x,\qquad
p_y = o_x\sin\theta+o_y\cos\theta+c_y .
\]

In a mathematical frame (\(y\) up) this order is counter-clockwise and the shoelace sum is positive (a side-2 square sums to 8, area 4). In an image frame (\(y\) down) it looks clockwise — **the direction itself does not matter, only that both polygons use the same one**, because the inside/outside test depends on consistently oriented edges. Mixing windings (A counter-clockwise, B clockwise) flips "inside" for one polygon's edges and the intersection becomes empty or a wrong polygon.

**Clipping.** Clip A's polygon by B's four directed edges \( a\to b \) with \( e=(b_x-a_x,\ b_y-a_y) \), one edge at a time:

\[
s(p)=e_x(p_y-a_y)-e_y(p_x-a_x)
\]

is the cross-product sign of \(p\) relative to that edge's line (the inner side of a counter-clockwise polygon is \(s\ge0\)). For each polygon edge \( p\to q \) there are three cases: both ends inside keep \(q\); both outside keep nothing; one in one out inserts the intersection at

\[
t=\frac{s(p)}{s(p)-s(q)}
\]

and keeps \( p+t(q-p) \). \(t\) comes from solving \( s(p)+t\,(s(q)-s(p))=0 \) for the linear interpolation, and the denominator is non-zero exactly in the mixed case. The output of each pass feeds the next; after four edges the result is the intersection polygon. Points on an edge count as inside (\(s\ge0\)), so two boxes that merely share an edge produce area 0 rather than NaN.

**Area and IoU.** Shoelace:

\[
|P|=\frac12\left|\sum_i (x_i y_{i+1}-x_{i+1}y_i)\right| ,
\]

and a polygon with fewer than 3 vertices (empty or degenerated to a segment) has area 0. The intersection is then clamped as \( \min(|A\cap B|,|A|,|B|) \): floating-point error can make it marginally larger than the smaller box, and after clamping "two identical boxes" yield IoU exactly 1.0 instead of 0.9999999999999999. Union \( =|A|+|B|-|A\cap B| \); a zero union (both areas degenerate, already excluded by the \(w,h>0\) validation) maps to IoU 0.

**The matrix.** \( \mathrm{iou}[i,j] \) comes from clipping \(P_i\) against \(P_j \); the intersection is set-theoretic and therefore order-free, so the matrix is symmetric, and \( \mathrm{iou}[i,i] \) is written as 1 by definition (self-clipping only adds rounding noise). The matrix is independent of NMS and always returned in full.

**NMS.** Processing order is \( (-\text{score},\text{index}) \), which `argsort(-scores, stable=True)` gives directly (score descending, equal scores by ascending index). The current box is checked against **all kept boxes**: if any IoU \( >\text{thr} \) it is suppressed, otherwise kept. Hence

- an IoU exactly equal to the threshold is kept (the predicate is "all \( \le \)");
- a suppressed box never enters the kept set, so later boxes are not compared with it;
- in the chain case (A–B overlap 1/3, B–C overlap 1/3, A–C disjoint), B is suppressed by A, so C has no overlap with the kept set and must be kept — an implementation comparing only with "the previous processed box" drops C incorrectly.

**dtype.** All geometry runs in float64: float32 inputs are promoted with `boxes.to(torch.float64)`, and the returned matrix is always float64; `keep`/`keep_mask` are dtype-independent.

## Key points and common mistakes

1. **Corner order is the docstring's**: the offset order and the direction of \( \theta \) are part of the contract; treating \( \theta \) as clockwise or swapping w/h silently ruins every IoU without any shape error.
2. **Angles are radians**: DOTA-style datasets often ship degrees; feeding those to `cos/sin` yields a completely different box.
3. **Both polygons must share a winding**: the inside test relies on directed edges; inconsistent windings corrupt the intersection.
4. **Clip edge by edge, iteratively**: feed each pass's output into the next; computing four half-plane intersections separately and stitching them is not the same thing.
5. **The intersection parameter \(t\) comes from the cross-product interpolation**: approximating with midpoints or nearest vertices shows visible error on oblique 45° intersections.
6. **Use rotated IoU, never the AABB**: two diagonally berthed ships can have AABBs overlapping at 0.6 while the rotated boxes overlap at 0.05 — the AABB deletes the correct box, which is the whole reason this problem exists.
7. **Only a strict `>` suppresses**: the keep predicate is \( \le\text{thr} \); writing \( < \) deletes a box whose IoU equals the threshold (the IoU=1/3, thr=1/3 case).
8. **Compare against all kept boxes**: comparing only with the last one drops an extra box in the chain case (`[0]` instead of `[0,2]`).
9. **Ties go to the smaller index**: a stable sort guarantees it; an unstable `argsort`, or re-ordering after sorting, changes the processing order and possibly the kept set.
10. **Clamp the intersection area**: without clamping, identical boxes can produce 1.0000000000000002, failing `torch.equal` or a range assertion; clamping also removes tiny negatives.
11. **Set the diagonal to 1 and keep the matrix symmetric**: self-clipping drifts from 1 by rounding; the matrix is judged, so it must match the definition.
12. **Empty input**: \(N=0\) returns `(0,)` int64, `(0,)` bool and `(0,0)` float64 instead of raising.
13. **Validation**: dtype must be float32/float64 and consistent between boxes and scores, \(w,h>0\), no NaN/Inf, and `iou_thr` in \( [0,1] \); all are in the contract — `assert` or silently producing NaN is wrong.
14. **Do not modify inputs**: `to(float64)` allocates a new tensor; no in-place op is allowed.

## Differences from neighboring problems

- **vs `nms`**: that one is axis-aligned hard NMS on \([x_1,y_1,x_2,y_2]\) boxes, IoU has a closed form, and only `keep` is returned; this one takes \( (c_x,c_y,w,h,\theta) \), requires polygon clipping, and also returns the full matrix and a mask.
- **vs `soft-nms`**: Soft-NMS continuously decays scores with a kernel instead of deleting boxes; this problem is hard suppression, but the geometry moves from axis-aligned to rotated, so the difficulty is the overlap area rather than the decay kernel.
- **vs `iou-compute` / `iou-variant-losses`**: those IoUs are axis-aligned (plus GIoU/DIoU/CIoU variants), all with closed forms; rotated IoU has none and must be clipped.
- **vs `vision-wbf-fusion` (same batch)**: WBF also walks boxes by descending score greedily, but it **fuses** (confidence-weighted coordinate averaging) whereas this problem **suppresses** (keep one, drop the rest).
- **vs `vision-map-coco` / `vision-anchor-match` (same batch)**: there IoU decides TP/FP or label assignment (matching); here it drives post-processing deduplication, and those boxes are axis-aligned.
- **vs `vision-fpn-levels` (same batch)**: FPN produces multi-scale features and an anchor ledger; this problem consumes the detection head's box outputs. They are adjacent in the pipeline but share no computation.

## Cross-check convention and complexity

- **Main oracle**: `_oracle` builds corners in pure Python, implements the same clipping with the **opposite traversal** (collecting on exiting \(p\to q\), while the reference collects on entering), sums the shoelace in Python loops, and runs NMS with `sorted(range(n), key=(-score, index))`; the reference uses tensorised clipping and `argsort(stable=True)`. It covers 3 configs (\(n=6,5,4\), including rotated copies sharing a centre) × 2 thresholds (0.3, 0.5), comparing the matrix with `atol=1e-9` and `keep`/`mask` exactly.
- **Independent analytic checks**: at \( \theta=0 \) the result must equal the closed-form axis-aligned rectangle IoU (20 random pairs, `atol=1e-12`), and rotating both boxes by the same \( \phi \) must leave the IoU unchanged (8 cases, `atol=1e-9`). Neither path touches the clipper, so they catch a clipper that is wrong yet self-consistent.
- **Discriminating cases (measured)**:
  - Example: a \(2\times2\) square and its \(45^\circ\) rotation about the same centre; intersection \(8(\sqrt2-1)=3.3137084989847603\), union \(16-8\sqrt2=4.68629150101524\), IoU \(=1/\sqrt2=0.7071067811865476\); `keep=[0,2]`, `keep_mask=[True,False,True]`; with threshold 0.71 the result becomes `keep=[0,1,2]`.
  - A \(4\times2\) box rotated \(90^\circ\) about its centre: intersection \(2\times2=4\), union \(8+8-4=12\), IoU exactly \(1/3\).
  - A \(2\times2\) box fully inside \(10\times10\): IoU \(=4/100=0.04\) (under containment IoU is the area ratio, independent of translation).
  - Identical boxes (including \(0.37\) rad): IoU exactly 1.0 (through the clamp); disjoint boxes: exactly 0.
  - Score ties: with scores \( [0.5,0.9,0.9,0.7] \) the order is 1, 2, 3, 0; box 0 coincides with box 2 and is suppressed last, giving `keep=[1,2,3]` and `mask=[False,True,True,True]`.
  - Chain: \(10\times10\) boxes centred at \(x=5,10,15\) give adjacent IoU \(=1/3\) and 0 between the ends; with `thr=0.3`, `keep=[0,2]`.
  - Threshold boundaries: at IoU \(=1/3\), `thr=1/3` keeps both while `thr=0.3` keeps one; at IoU \(=1\), `thr=1.0` keeps both while `thr=0.99` keeps one; disjoint boxes are both kept at `thr=0`.
- **Non-obvious measured facts**: two squares rotated by \(45^\circ\) share 82.8% of their area yet have IoU only 0.707 — area fraction and IoU are different quantities; \(1/3\) and \(4/12\) are the same correctly rounded double, so the "equal to threshold is kept" assertion is stable; float32 inputs promoted to float64 agree with float64 inputs within \(10^{-6}\) on the matrix and produce identical `keep`.
- **Complexity**: all pairs are clipped, each clip \(O(V+E)=O(8+4)\) vertex operations, so \(O(N^2)\) polygon operations, plus \(O(NK)\le O(N^2)\) IoU lookups in NMS. Judge scale is \(N\le8\), far below any limit; production `box_iou_rotated` tensorises the clipping on GPU and prunes candidates with sorting/grids instead of a full \(N^2\).
