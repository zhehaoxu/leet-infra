# In Depth: Detection: Deformable Sampling and Aggregation

## Source

- Deformable convolution: Dai et al., *Deformable Convolutional Networks*, ICCV 2017. Each kernel sampling location regresses an extra offset \( \Delta p \) and reads the value at \( p+\Delta p \) by bilinear interpolation; the offset is learned end-to-end through the bilinear weights, and out-of-bounds positions are treated as 0.
- Deformable attention: Zhu et al., *Deformable DETR: Deformable Transformers for End-to-End Object Detection*, ICLR 2021. Each query's reference point is augmented with a set of learned offsets, values are sampled from single- or multi-scale features, and the sampled values are summed with softmax attention weights.
- Engineering references: `torch.nn.functional.grid_sample` (with `mode="bilinear"`, `padding_mode="zeros"`, `align_corners=True` it matches this problem's convention once coordinates are normalized), torchvision's `deform_conv2d`, and mmcv's `MSDeformAttn`. None is a dependency here; the judge only has torch.
- CPU only, deterministic, no RNG: only floor, arithmetic and indexing.
- This problem isolates the "single-scale, single-head, single-level" core: the input is a set of arbitrary pixel coordinates plus offsets, the output the per-point sampled values and an optional weighted aggregation; multi-scale, multi-head and softmax stay outside the boundary.

## Full derivation and index convention

**Sampling position.** The \(k\)-th point of batch element \(n\) sits at

\[
x_{nk}=p^x_{nk}+d^x_{nk},\qquad y_{nk}=p^y_{nk}+d^y_{nk},
\]

where \(p\) is `points` and \(d\) is `offsets`. Coordinates are **pixel indices**, not normalized: \(x\) is the column, \(y\) the row, \( (0,0) \) is the top-left pixel and \( (W-1,H-1) \) the bottom-right one.

**Bilinear.** With \(x_0=\lfloor x\rfloor\), \(y_0=\lfloor y\rfloor\), \(f_x=x-x_0\), \(f_y=y-y_0\),

\[
I(x,y)=(1-f_x)(1-f_y)\,F[y_0,x_0]+f_x(1-f_y)\,F[y_0,x_0+1]
+(1-f_x)f_y\,F[y_0+1,x_0]+f_xf_y\,F[y_0+1,x_0+1],
\]

where \(F\) is 0 outside \( [0,W-1]\times[0,H-1] \) (per-corner zero padding). The four weights are non-negative and sum to

\[
(1-f_x)(1-f_y)+f_x(1-f_y)+(1-f_x)f_y+f_xf_y
=(1-f_y)\big[(1-f_x)+f_x\big]+f_y\big[(1-f_x)+f_x\big]=1,
\]

so with all four corners inside the result is a convex combination and cannot overshoot; near the border some weights fall outside and become 0, which just removes part of the contribution — still non-negative for non-negative features, but no longer a convex combination.

**Floor and negatives.** `floor` must really round down: \( \lfloor-0.5\rfloor=-1 \), hence \(f_x=0.5\), with weight 0.5 on the in-range corner \(x=0\) and 0.5 on the out-of-range corner \(x=-1\), so only half survives. Using `int(-0.5) == 0` (truncation toward zero) gives \(x_0=0,f_x=-0.5\), i.e. weights \(1.5\) and \(-0.5\) — out of range and negative; `round` snaps half-pixel positions onto integer pixels. All three produce different numbers.

**Continuity at the border.** Zero-padding sampling is continuous in the coordinate: as \(x\to-1^+\), \(x_0=-1\) and \(f_x\to0^+\), the surviving corner \(x=0\) has weight \(f_x\to0\), so the result tends to 0, matching the result at exactly \(x=-1\); as \(x\to0^+\) the result tends to \(F[0,y]\). The **derivative** jumps at integers and at the border, which is why bilinear sampling is piecewise linear on a coarse grid and part of why DCN regularises offset learning.

**Exact integer corners.** When \(x\) is an integer, \(f_x=0\) and the two \(x_0+1\) corners carry weight 0; even if they are out of range (e.g. \(x=W-1\) makes \(x_0+1=W\)) they cannot affect the result. An implementation therefore must not raise or truncate just because an index is out of range — the right move is to compute the four integer corner coordinates and zero the out-of-range contributions with a validity mask.

**A robust vectorised gather**: clamp the corner coordinates into \( [0,W-1]/[0,H-1] \), cast to int64 and gather, then multiply by the validity mask. Clamping only touches terms that will be zeroed anyway, and it avoids int64 overflow for absurdly large coordinates; the order (floor → compare → clamp → cast) matters.

**Aggregation.** Given \( (N,K) \) weights \(w\),

\[
\mathrm{out}[n,c]=\sum_{k=0}^{K-1}w[n,k]\,\mathrm{sampled}[n,k,c],
\]

**without** dividing by \( \sum_k w[n,k] \). Deformable DETR's weights already come from a softmax (normalized over the \(K\) points, possibly across levels and heads); normalizing again inside the op changes the semantics. Weights may be negative, may be all zero (result all zero), and may sum to anything. Treat \(K=1, w=2 \) as "twice the sampled value", not "the same value".

**dtype.** Everything is promoted to float64 internally and cast back with `to(feature.dtype)` at the end; for float32 inputs the rounding happens once, which is closer to the true value than accumulating in float32 (measured maximum difference between the two dtypes on random data: about \( 4\times10^{-7} \)).

## Key points and common mistakes

1. **The sampling position is point + offset**: using only `points` or only `offsets` is wrong, and the offset components are \( (\mathrm{d}x,\mathrm{d}y) \).
2. **x is the column, y the row**: index as `feature[:, y, x]`; treating \( (x,y) \) as \( (\text{row},\text{col}) \) silently returns the transposed feature.
3. **Out-of-range corners contribute 0, never a clamped edge**: clamping replicates border pixels, so \(x=-0.5\) yields `F[0,0]` instead of `0.5*F[0,0]`.
4. **Check each corner separately**: "the sample point is inside" does not mean all four corners are; the right neighbours of \(x=W-1\) are outside.
5. **floor is neither truncation nor rounding**: on negative coordinates `int()` produces the wrong fraction and negative weights.
6. **The weight order is fixed**: \( (1-f_x)(1-f_y),f_x(1-f_y),(1-f_x)f_y,f_xf_y \) pair with \( (x_0,y_0),(x_0+1,y_0),(x_0,y_0+1),(x_0+1,y_0+1) \); swapping \(f_x/f_y\) transposes the image.
7. **Aggregation is not normalized**: do not divide by the weight sum; all-zero weights must give 0, not NaN (avoid \( \sum w\cdot v/\sum w \)).
8. **Do not modify inputs**: coordinate addition allocates a new tensor; no in-place `[...]=` is allowed.
9. **The layout is \( (N,K,C) \)**: not \( (N,C,K) \) and not \( (K,N,C) \); only the weighted variant collapses to \( (N,C) \).
10. **\(N=0\) and \(K\ge1\)**: an empty batch returns \( (0,K,C) \)/\( (0,C) \) rather than raising; \(K=0\) raises `ValueError` per the contract.
11. **Shape/dtype validation**: `points` and `offsets` must share shape, have a last dimension of 2, and match `feature`'s dtype; the judge deliberately mixes float64 features with float32 offsets as an error case.
12. **NaN/Inf validation**: non-finite entries in `points`/`offsets`/`weights` must raise instead of propagating through `floor(NaN)`.
13. **Clamp before casting huge coordinates**: a direct `to(torch.int64)` overflows at values like \(10^{30}\); clamp to the valid range first.
14. **Do not drop to float32 midway**: accumulating four corners and \(K\) weighted terms in float32 amplifies error; the contract requires float64 internally.

## Differences from neighboring problems

- **vs `grid-sample`**: that problem uses **normalized** coordinates (\([-1,1]\) with an `align_corners=False` mapping), takes \( (N,C,H,W) \) plus a regular grid, and returns \( (N,C,H_o,W_o) \); this one uses **pixel** coordinates, takes an arbitrary point set with offsets, puts the point dimension last, and adds an optional weighted aggregation. The zero-padding convention is the same.
- **vs `roi_align` / `roi_pool`**: those lay a regular sampling grid inside each RoI bin and average (RoI Pool also quantizes), with positions determined by the RoI and `spatial_scale`; here positions come from learned offsets, can be arbitrary and grouped, and there is no averaging.
- **vs `upsample-bilinear` / `upsample-nearest`**: those upsample a whole image on a regular grid (alignment modes and scale factors are the point); this one samples sparse, irregular positions with no output grid.
- **vs `vision-fpn-levels` (same batch)**: FPN's upsampling is integer nearest-neighbour with no sub-pixel positions; deformable sampling is precisely sub-pixel.
- **vs `vision-anchor-match` / `vision-fcos-centerness` (same batch)**: those generate/assign anchors and labels and never interpolate values; this problem involves no boxes or matching.
- **vs `vision-rotated-nms` (same batch)**: that one computes polygonal overlap areas; this one reads from a feature map, with tensors on both sides rather than geometric objects.

## Cross-check convention and complexity

- **Main oracle**: `_oracle` expands every point and channel in pure Python: `math.floor` for the corners, an explicit per-corner in-range check that skips outside corners, accumulation in the **reverse order** \( (1,1)\to(1,0)\to(0,1)\to(0,0) \) (a different floating-point path from the reference's \( (0,0)\to(0,1)\to(1,0)\to(1,1) \)), and Python sums for the weighted aggregation; the reference uses tensor gathers plus a validity mask. It covers 4 configs (\(C=1..3\), growing \(H,W\), \(N=4,K=6\), coordinates reaching 2 pixels outside, offsets \( \pm1.5 \)), with and without weights, at `atol=1e-12` for float64 and `1e-5` for float32. The measured maximum difference between the two paths is \( 2.2\times10^{-16} \).
- **Discriminating cases (measured)**:
  - Example: `sampled = [[5.5,55],[0.5,5],[7,70]]`, aggregated with \( [0.5,0.25,0.25] \) into \( [4.625,46.25] \).
  - Integer coordinates with zero offsets reproduce the pixels exactly, and \( (x=2,y=1) \) must be `6/60` (column/row not mixed).
  - Zero padding: \( (-0.5,0)\to0.5\cdot F[0,0] \), \( (-1.0,0)\to0 \), \( (-0.5,0.5)\to0.25(F[0,0]+F[1,0]) \), \( (2.5,0)\to0.5\cdot F[0,2] \), \( (2.5,2.5)\to0.25\cdot F[2,2] \), \( (50,-20)\to0 \).
  - Offsets: \( (0,0)+(2,1)\to F[1,2]=6/60 \) (ignoring the offset gives \(1/10\)); \( (0,0)+(-0.5,-0.5)\to0.25\cdot F[0,0] \).
  - Weights: \( [1,1,0] \) gives \(6/60\) (normalized would give \(3/30\)); all zero gives 0; \(K=1,w=2\) doubles the sample; negative \( [1,-1] \) gives the difference of the two points.
  - \(N=0\) returns \( (0,2,2) \) and \( (0,2) \); float64 and float32 inputs agree within \(10^{-5}\) (measured about \(4\times10^{-7}\) on random data); 18 invalid-input cases.
- **Extra independent check (measured)**: converting coordinates to an `align_corners=True` normalized grid and calling `F.grid_sample(mode="bilinear", padding_mode="zeros")` differs from this implementation by at most \( 1.7\times10^{-16} \) — the zero-padding and bilinear conventions match PyTorch's official op rather than a bespoke border rule.
- **Complexity**: four corner reads and a handful of multiply-adds per point, i.e. \( O(N\cdot K\cdot C) \) memory accesses and arithmetic; judge scale is \(N\le4,K\le6,C\le3\), far below any limit. In production Deformable DETR batches the \( (N,K,C) \) gather into one indexing pass, with \(K\) typically 4 per level times the number of levels.
