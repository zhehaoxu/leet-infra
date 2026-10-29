# Detection: Rotated-Box NMS and Overlap Area

## Problem

Aerial imagery, remote sensing and scene text have no notion of "horizontal": a berthed ship, a tilted shelf row or a curved text line all look wrong inside an axis-aligned bounding box (AABB), which drags in a lot of background. The industry answer is to regress **rotated boxes** \( (c_x,c_y,w,h,\theta) \) directly and suppress with rotated IoU. Axis-aligned NMS breaks down here: two ships lying side by side along a diagonal can have AABBs overlapping at 0.6 while the real rotated boxes overlap by 0.05 — suppressing on the AABB deletes the correct detection.

Rotated IoU has no closed form: two arbitrarily rotated rectangles intersect in a convex polygon with at most eight edges, so the geometry has to be **computed**. This problem asks for building four corners from centre/size/angle, clipping one polygon by the other (Sutherland–Hodgman) to get the intersection, taking the area with the shoelace formula, producing the full \(N\times N\) IoU matrix, and then running greedy NMS ordered by (score descending, smaller index first). Three things go wrong easily:

- **Corner winding**: clipping needs the clip polygon counter-clockwise; reversing it flips every inside/outside test;
- **Threshold strictness**: a box whose IoU equals the threshold is kept; only strictly greater suppresses;
- **Comparison target**: a box must be compared with **all kept boxes**, not just the last one.

CPU only, fully deterministic, no RNG.

## Rules

The complete signature, parameter meanings and defaults live only in the starter docstring, which is the sole judging contract.

- Each box is \( (c_x,c_y,w,h,\theta) \): centre, width, height and a **counter-clockwise** angle in radians. The four corners sit at offsets \( (+w/2,-h/2) \), \( (+w/2,+h/2) \), \( (-w/2,+h/2) \), \( (-w/2,-h/2) \) in the centre frame (counter-clockwise), then the whole box is rotated by \( \theta \) and translated. \(w>0\), \(h>0\); \(N\) may be 0.
- IoU: the intersection area comes from clipping A's quadrilateral against B's four edges with Sutherland–Hodgman, area by the shoelace formula; the union is \( |A|+|B|-|A\cap B| \) and IoU \( = \) intersection / union. All geometry is done in float64.
- Return the full \( (N,N) \) float64 IoU matrix: diagonal 1, symmetric, entries in \( [0,1] \). It is independent of the NMS outcome and always returned in full (the matrix is itself judged).
- NMS: process boxes by (score descending, ties by original index ascending); box \(i\) is kept iff \( \mathrm{IoU}(i,k)\le \text{thr} \) for **every kept** box \(k\). Only a strict `>` suppresses, so an IoU exactly at the threshold is kept; a suppressed box takes no further part in later comparisons.
- Return `(keep, keep_mask, iou_matrix)`: `keep` is int64 \( (K,) \) of original indices in processing order (score descending); `keep_mask` is bool \( (N,) \) with True exactly at `keep`; `iou_matrix` is float64 \( (N,N) \). For \(N=0\) the three are \( (0,) \), \( (0,) \), \( (0,0) \).
- Invalid input raises `ValueError`: not a tensor, boxes not \( (N,5) \), scores not 1-D or with the wrong length, dtype not float32/float64 or mismatched between boxes and scores, \(w\le0\) or \(h\le0\), NaN/Inf, `iou_thr` outside \( [0,1] \).

## Example

Three boxes, scores \( [0.9,0.8,0.7] \), threshold 0.5:

```
A = (0, 0, 2, 2, 0)          axis-aligned 2x2 square centred at the origin
B = (0, 0, 2, 2, pi/4)       the same square rotated 45 degrees counter-clockwise
C = (10, 10, 2, 2, 0)        far away, disjoint from A and B
```

The intersection of A and B is a regular octagon. View B as the diamond \( |x+y|\le\sqrt2,\ |x-y|\le\sqrt2 \) (a square of side 2 has vertices at \( (\pm\sqrt2,0),(0,\pm\sqrt2) \)). Relative to \( [-1,1]^2 \) only the four corners \( (\pm1,\pm1) \) are cut off: at the corner \( (1,1) \), the line \( x+y=\sqrt2 \) meets \( x=1 \) and \( y=1 \) at \( (1,\sqrt2-1) \) and \( (\sqrt2-1,1) \), so the removed isosceles right triangle has legs \( 2-\sqrt2 \) and area \( \tfrac12(2-\sqrt2)^2 \). Hence

\[
|A\cap B| = 4-4\cdot\tfrac12\,(2-\sqrt2)^2 = 8(\sqrt2-1)\approx 3.313708 ,
\]

the union is \( 4+4-3.313708=16-8\sqrt2\approx 4.686292 \), and

\[
\mathrm{IoU}(A,B)=\frac{8(\sqrt2-1)}{16-8\sqrt2}=\frac{\sqrt2-1}{2-\sqrt2}=\frac{1}{\sqrt2}\approx 0.707107 .
\]

```
IoU(A, B) = 0.7071067811865476   (measured, = 1/sqrt(2))
IoU(A, C) = 0
IoU(B, C) = 0
```

NMS: A (0.9) is kept; B (0.8) has IoU 0.7071 > 0.5 with A and is suppressed; C (0.7) has IoU 0 with A and is kept.

```
keep      = [0, 2]
keep_mask = [True, False, True]
```

Note the unit diagonal (self-comparison) and the symmetry of the matrix; change the threshold to 0.71 and B's 0.7071 no longer exceeds it, so `keep` becomes `[0, 1, 2]` — exactly the discriminator for "strictly greater suppresses".

## Constraints

- Return `(keep, keep_mask, iou_matrix)`; the matrix is float64 regardless of the boxes' input dtype; `keep` is int64 and `keep_mask` is bool.
- The judge uses absolute tolerance \( 10^{-9} \) on the matrix for float64 input (or \( 10^{-6} \) for float32 input) and `torch.equal` for `keep`/`keep_mask`.
- Corner order, counter-clockwise angles, intersection clipping, the union identity, threshold strictness, index tie-breaking, "compare against all kept boxes", and the diagonal/symmetry of the matrix are all part of the contract.
- CPU only, deterministic, no RNG; \(N=0\), disjoint boxes, fully coincident boxes and non-axis-aligned angles must all return the described results rather than raising.
