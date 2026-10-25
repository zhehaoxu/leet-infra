# RoI Align

## Problem Description

RoI Align (introduced by Mask R-CNN) fixes the **quantization error** of RoI Pooling: no rounding
is performed at all; both ROI and bin boundaries stay floating-point; for each output bin a
**regular $s \times s$ sampling grid** is placed within its feature-coordinate range, each sampling
point is read from the feature map via **bilinear interpolation**, and finally all sampling points
in the grid are **averaged**. This makes the output sub-pixel aligned with the ROI, giving markedly
better accuracy than RoI Pooling for small objects / mask prediction.

## Rules

- Implement `solve(features, rois, pooled_size, spatial_scale, sampling_ratio, aligned)`.
  - `features` has shape `(N, C, H, W)`; `rois` has shape `(K, 5)`, each row `[batch_index, x1, y1, x2, y2]` (float coordinates in the original image).
  - `pooled_size` $P$ (output is $P\times P$); `spatial_scale` $s$ is the original-image→feature-map scale; `sampling_ratio` $r\ge1$ is the number of sampling points per bin per axis; `aligned` is a boolean.
- For each ROI:
  1. Mapping (**no quantization**): $x_1 = X_1 s,\ y_1 = Y_1 s,\ x_2 = X_2 s,\ y_2 = Y_2 s$; if `aligned` is true, subtract $0.5$ from each of the four values.
  2. Bin size: $\text{bin}_w=(x_2-x_1)/P,\ \text{bin}_h=(y_2-y_1)/P$.
  3. Floating-point range of bin $(i,j)$: $[\,x_1+j\,\text{bin}_w,\ x_1+(j+1)\text{bin}_w)$ (same for $y$).
  4. Place an $r\times r$ grid inside the bin; the feature coordinates of sampling point $(a,b)$:
     $$ f_x = x_{start} + (b+0.5)\frac{\text{bin}_w}{r},\qquad f_y = y_{start} + (a+0.5)\frac{\text{bin}_h}{r} $$
  5. Each sampling point is **bilinearly interpolated** per channel: take $\lfloor f_x\rfloor,\lfloor f_y\rfloor$ and their +1 neighbors, forming the 4-point neighborhood with distance weights;
     **grid points falling outside the feature map contribute 0** (no clamp-to-edge). Bin output = the **average** of the $r^2$ sampling points.
- Output shape `(K, C, P, P)` (accumulating in float64 is recommended to match the reference precision).

> torchvision is not available in this environment; correctness is judged against an
> **independent numpy reference implementation** (the judge verifies with a separate numpy implementation).
> Implement strictly per the rules above: "no quantization / aligned offset / grid sampling / out-of-bounds counts as 0 / average".

## Example

```
Input: features = arange(16).reshape(1,1,4,4)      # values 0..15
     rois = [[0, 0, 0, 3, 3]], pooled_size=2, spatial_scale=1.0
     sampling_ratio=2, aligned=False
   ROI feature range [0,3]×[0,3], bin_w=bin_h=1.5; each bin gets a 2×2 grid, bilinear then averaged
Output (float64): [[[[3.75, 5.25],
                   [9.75, 11.25]]]]
```

## Constraints

- float32 input; `pooled_size >= 1`, `sampling_ratio >= 1`, `spatial_scale > 0`
- When a sampling point is out of bounds, its out-of-bounds neighborhood grid points contribute 0 (equivalent to evaluating bilinear interpolation on a zero-padded feature map)
- Do not use `torchvision.ops.roi_align` / `F.grid_sample`; hand-write the grid sampling + bilinear interpolation
