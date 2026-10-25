# RoI Align Explained

## What It Improves over RoI Pooling

RoI Pooling has two irreversible roundings: ROI coordinates are `floor`ed to integer grid points, and bin boundaries use integer division. For small ROIs, these few pixels of displacement make the sampled features misaligned with the box's true position, and max amplifies this noise. The core of RoI Align is
**no quantization anywhere**:

1. ROIs and bin boundaries are all floating-point;
2. Each bin does not take the "window max" but instead places an $r\times r$ grid, applies **bilinear interpolation** at each sampling point and then takes the **average**.

Sampling points fall between four integer grid points and are combined by distance weights $\Rightarrow$ sub-pixel precision, differentiable, and smooth under position perturbations.

## Meaning of aligned

`aligned=True` corresponds to the `align_corners=True` coordinate convention: after mapping the ROI, subtract $0.5$ from each of the four values so that feature grid-point centers align with pixel centers; `aligned=False` does not subtract. Both conventions appear in practice; this problem exposes `aligned` as an explicit parameter and the judge implements the same rule to avoid ambiguity.

## Out-of-Bounds Counts as 0

This implementation specifies that grid points of a sampling point's 4-neighborhood that
**fall outside the feature map contribute 0** (equivalent to zero-padding the feature map and then evaluating bilinear interpolation).
This differs from "clamp to the edge first, then interpolate"; it is a simplifying convention for torchvision RoIAlign's out-of-bounds behavior; since torchvision is unavailable in the environment, correctness follows this rule in the problem statement plus an independent numpy implementation.

## Implementation Points

- Accumulate in **float64**: interpolation/averaging involves many fractional values, and float32 accumulated error may exceed `atol`.
- Grid sampling center: `f = start + (idx + 0.5) * bin / r`; note the `+0.5` takes each sub-cell's center.
- Bilinear: `y0=floor(fy), x0=floor(fx)`, combine the 4 neighbors with weights such as `(1-ly)(1-lx)`, out-of-bounds terms return 0.

## Common Pitfalls

- Forgetting to multiply by `spatial_scale`, or using an adaptive `ceil(bin)` branch for `sampling_ratio` (this problem requires a fixed `r`).
- Dividing by `r*r` when averaging (not by the number of valid points sampled).
- The `aligned` branch applies `-0.5` only to the ROI endpoints; do not subtract again at the sampling points.

## Applications

Mask R-CNN uses RoI Align to produce mask features precisely aligned with the box, which is key to instance segmentation; detection/segmentation/pose and other tasks needing pixel-level localization have almost all migrated from RoI Pooling to RoI Align. In multi-task settings, different heads can each use a different `P`.
