# RoI Pooling Explained

## Why It Is Needed

Two-stage detectors (Faster R-CNN) first have the RPN propose many candidate boxes of **arbitrary size**, but the fully-connected head behind requires a **fixed-length** feature vector. RoI Pooling is responsible for "squashing" each ROI on the feature map into a uniform
$P\times P$, so it can feed the shared classification/regression heads.

## Quantization Is Its Original Sin

The two roundings in RoI Pooling introduce **position quantization error**:

1. ROI coordinates are `floor`ed to integer grid points after multiplying by `spatial_scale`;
2. Each bin boundary is further discretized with **integer division**.

For large ROIs this error is negligible, but on small objects a few pixels of offset misalign the features from the box and hurt localization accuracy,
and max pooling is very sensitive to this misalignment. In addition, during the backward pass gradients can only be routed to the grid point that attained the max, and quantization makes
forward/backward coordinate alignment more fragile.

## Comparison with RoI Align

| | RoI Pooling | RoI Align |
|---|---|---|
| Coordinate quantization | floor / integer-division rounding | keeps floats |
| Bin aggregation | max over an integer window | regular sampling points + bilinear interpolation then **average** |
| Small objects | noticeable error | sub-pixel alignment |

RoI Align (see another problem in this collection) was born precisely to eliminate the quantization error: no rounding; for each bin it uses `sampling_ratio`
sampling points, applies bilinear interpolation, then averages.

## Common Pitfalls

- **Bin boundaries use floor division**: `ylo = fy1 + (fy2-fy1)*i // P`, exactly matching the reference numpy; using float interpolation turns it into
  an Align flavor that will not match.
- **Empty bin → 0**: when the ROI is too small, multiple bins land on the same cell, or clipping out-of-bounds leaves a bin empty, output 0 instead of raising an error.
- **Clip to boundaries**: `max(0,·)`, `min(H/W,·)` to prevent out-of-bounds indexing.
- **batch_index** is the first column of the ROI, used to select the image along the N dimension of `features`.

## History

SPPnet first used spatial pyramid pooling to connect arbitrary-size inputs with a fixed FC; Fast RCNN simplified it to single-scale RoI Pooling;
Faster RCNN kept it and hooked it up to the RPN; Mask RCNN identified the quantization error problem and proposed RoI Align. This environment has no torchvision,
so implement it by hand in pure torch and cross-validate correctness with an independent numpy implementation.
