# In Depth: Vision Detection: FCOS Center-ness and Multi-Level Assignment

## Source

- Point-wise detection and center-ness: Tian et al., *FCOS: Fully Convolutional One-Stage Object Detection*, arXiv:1904.01355 (ICCV 2019). Sec. 3.1 introduces the \(l,t,r,b\) regression and \( \mathrm{centerness}=\sqrt{\frac{\min(l,r)}{\max(l,r)}\cdot\frac{\min(t,b)}{\max(t,b)}} \) (written \( \sqrt{\frac{\min(l^*,r^*)}{\max(l^*,r^*)}\cdot\frac{\min(t^*,b^*)}{\max(t^*,b^*)}} \) there), multiplied into the classification loss to down-weight poor locations.
- Multi-level assignment (regress range): Sec. 3.2 of the same paper — each level owns objects whose \( \max(l^*,t^*,r^*,b^*) \) lies in \( (m_{i-1},m_i] \), and locations outside the range receive no loss at all (the ignore class here). The paper uses \(m=\{0,64,128,256,512,\infty\}\); this problem exposes it as `regress_ranges` with a left-closed/right-open convention.
- Center sampling: Sec. 3.2 and mmdetection's `FCOSHead` with `center_sampling_radius=1.5` — the centre region is a square of half-side \(r\times\text{stride}\) around the GT centre, not the box shrunk inwards.
- Ambiguity resolution for locations inside several boxes: pick the smallest-area GT, as in the FCOS paper.
- CPU only and fully deterministic: arithmetic on integer grid coordinates plus one square root; no RNG, no images read.

## Full derivation and index convention

**Point-wise distances can be negative.** They are

\[
l=x-x_1,\quad t=y-y_1,\quad r=x_2-x,\quad b=y_2-y,
\]

so some are negative outside the box. The reference broadcasts four \( (P,G) \) matrices:

```
l = px[:, None] - x1[None, :]      # (P,G)
t = py[:, None] - y1[None, :]
r = x2[None, :] - px[:, None]
b = y2[None, :] - py[:, None]
```

`inside = (l>=0)&(t>=0)&(r>=0)&(b>=0)` is inclusive, so a location sitting exactly on a box edge counts as inside — the boundary locations \((2,12)\) and \((22,12)\) in the tests rely on this (center-ness 0 but still trained).

**Center sampling is a square, not a shrunk box.** The radius is \( \rho=\text{center\_radius}\times\text{stride}_{\text{point's level}} \) and the test is

\[
|x-c_x|\le\rho,\qquad |y-c_y|\le\rho,\qquad c_x=\frac{x_1+x_2}{2},\ c_y=\frac{y_1+y_2}{2}.
\]

It is **not** "inset each box side by \( \rho \)": for long thin boxes the two differ a lot (insetting would drop the whole centre line). mmdet implements the centre region exactly as a square around the centre. Because the radius depends on the point's level, the implementation first looks `strides` and `regress_ranges` up per point into \( (P,) \) vectors and then `unsqueeze(1)`-broadcasts them to \( (P,G) \).

**The scale interval is left-closed, right-open.** `in_range` uses **that GT's own** \( \max(l,t,r,b) \): the same location has a different max for different GTs, so "legal at this level" is a per-GT decision — you cannot compute one global scale and filter GTs afterwards. In the example, narrowing \([0,64)\) to \([0,16)\) pushes the first two locations (max 18) out → ignore; \([0,18)\) still excludes them because 18 is the open end; only \([0,20)\) makes them positive. The left end is inclusive: with \([18,40)\) the two max=18 locations stay legal.

**Why out-of-range means "ignore", not "negative".** Those locations are demonstrably on the object; the current level simply does not own that scale. Treating them as negatives teaches the classifier that the object's interior is background. The example's `[-2,-2,-2,0,-1]` versus `[-1,-1,-1,0,-1]` differ in two label codes but entirely in training semantics, and judging uses `torch.equal`. Implement it as two layers: `cand = inside & in_center & in_range` yields positives, `covered = inside.any(dim=1)` separates ignore from negative:

\[
\text{label}=\begin{cases}
\arg\min_{\text{cand}}\text{area}, & \text{has candidate}\\
-2, & \text{no candidate but covered}\\
-1, & \text{otherwise}
\end{cases}
\]

**Vectorising "smallest area".** Put `-area` at candidates and `finfo.min` elsewhere, then `max(dim=1)`: `torch.max` returns the **first** index on ties, which is exactly "smallest GT index on equal area". With `topk`/`sort` you must enforce stability yourself. In the measured nested case (big \(30\times30\) area 900, small \(8\times8\) area 64, location \((12,12)\) inside both) the small GT wins regardless of GT order, with center-ness 1.0 and regression target \([4,4,4,4]\).

**The zero convention of center-ness.** The formula hits 0/0 when \( \max(l,r)=0 \) (location on a left/right edge) or \( \max(t,b)=0 \) (top/bottom edge). The convention here is to take that factor as 0, so all four edges score 0 and only the box centre scores 1.0. Use `torch.where(max>0, min/max, 0)` — both branches of `torch.where` are evaluated, so a bare division leaves NaN. Degenerate GTs (\(x_1=x_2\) and \(y_1=y_2\), area 0) are handled the same way: area 0 wins the smallest-area contest, all four distances are 0, center-ness is 0, regression target is all zeros.

**dtype.** Center-ness and regression targets follow `gt_boxes`; in float32 \( \sqrt{2/18} \) differs from float64 by roughly \(10^{-8}\), hence tolerances \(10^{-6}\) (float32) / \(10^{-12}\) (float64). `labels` is always int64.

## Key points and common mistakes

1. **`in_range` is a per-GT test**: the max uses that GT's four distances, not a "point scale". Filtering GTs after computing one global scale kills locations that should be legal.
2. **Out-of-range is -2, not -1**: `inside` and `cand` are different boolean layers; compute both and combine.
3. **Center sampling is a square**: half-side `center_radius × stride` around the GT centre, not an inward shrink. With `center_radius=0` only a location exactly at the box centre is positive (measured `[-2, 0]`).
4. **Distances may be negative**: do not `clamp(min=0)` before `inside`, or outside points will look inside.
5. **Smallest area with ties to the smallest index**: `argmax`'s "first maximum" is exactly the required behaviour, but `sort`/`topk` need explicit stability; the duplicate-GT case pins index 0.
6. **The 0/0 in center-ness**: a factor with `max=0` is 0, and `where` must guard the division to keep NaN out of the output.
7. **Empty GT and empty points**: \(G=0\) → all -1; \(P=0\) → shapes \((0,)\), \((0,)\), \((0,4)\), and an empty `levels` must not be rejected as an error.
8. **Validation order**: check level bounds before building the lookup table, otherwise the list indexing raises `IndexError` instead of `ValueError`.
9. **`points` dtype must equal `gt_boxes`**: mixing float32/float64 would silently promote in the broadcast; the contract says raise `ValueError`.

## Differences from neighboring problems

- **vs `vision-anchor-match` (same batch)**: that is anchor-based — tile boxes, then use IoU for positives/negatives and the guarantee; this is anchor-free, with positives decided by "inside + centre region + scale range" and no IoU at all, and it additionally emits a center-ness soft label.
- **vs `vision-fpn-levels` (same batch)**: that problem builds the FPN neck (1×1 laterals, 2× top-down upsampling, 3×3 outputs) and keeps the level-size ledger — feature-map movement; this one only consumes "which level a point belongs to".
- **vs `vision-wbf-fusion` (same batch)**: WBF fuses scored detections after inference; this assigns labels to raw GTs and grid locations before training.
- **vs `vision-deformable-sample` (same batch)**: deformable sampling does bilinear reads at offset locations on a feature map; the "sample points" here are feature-grid centres at exact coordinates, with no interpolation.
- **vs `iou-variant-losses` (existing)**: that problem evaluates generalized IoU between two given boxes and never assigns labels; no IoU appears here.

## Cross-check convention and complexity

- **Main oracle**: `_oracle` is a pure-Python triple loop (point → GT → compare) that maintains `covered` and a `(area, gi)` best key explicitly and computes center-ness separately; the reference uses \( (P,G) \) broadcast matrices with `where` and `argmax`. The data structures and traversal are entirely different.
- **Coverage**: two levels with different strides and ranges, a single-level 4×4 grid, four non-overlapping small GTs (smallest-area and center sampling), `center_radius ∈ {0, 0.25, 5.0}`, nested GTs in both orders, duplicate GTs with equal area, centre-region ignore, four range-boundary variants, center-ness 0 on edges, degenerate GT, no GT, empty point set, float64, and 13 invalid-input cases.
- **Non-obvious measured facts**: the center-ness of location \((12,4)\) is \( \sqrt{2/18}\approx0.333333 \), not \(2/18\) — only the \(t,b\) pair differs, so the geometric mean differs from the arithmetic one; narrowing \([0,64)\) to \([0,16)\) changes only the first two labels; in the nested case "smaller max" and "smallest area" happen to agree, but the rules are not equivalent in general (area wins here).
- **Complexity**: \(O(PG)\) time and memory (four \( (P,G) \) distance matrices plus the candidate mask), plus \(O(P)\) vectors. Judge scale is \(P\le 16\), \(G\le 4\), far below the limit. Production code slices by level and keeps only candidate locations so that \(P\) in the hundreds of thousands does not blow up \( (P,G) \).
