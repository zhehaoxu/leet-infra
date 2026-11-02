# notes (vision-rfcn-psroi-pool)

## Source

- Paper: R-FCN (arXiv 1605.06409) §2.3. The position-sensitive score maps pair each of the
  `k×k` spatial cells with one of the `k²` channels; each bin average-pools only its own
  sub-region of the RoI, and the `k²` bin scores vote equally into the class score. The box
  branch uses the same scheme with `4k²` channels.
- **The implementation could not be pinned down**: the official repo `daijifeng001/R-FCN` keeps
  Caffe in the `external/caffe` submodule (`.gitmodules` points at `daijifeng001/caffe-rfcn`,
  branch `faster-R-CNN`), and that branch's tree comes back empty, so
  `psroi_pooling_layer.cu` is unavailable. This problem therefore **does not claim to replicate
  any kernel's float rounding**; the bin partition is normalized to integer division and written
  into the statement.
- As a consequence the bin edges may differ from the Caffe implementation by one pixel. That is
  the officially ambiguous part, not the subject of this problem.

## Normalizations in this problem

- Coordinates must be integer feature coordinates (floats raise `ValueError`); `spatial_scale`
  folding is left to the caller. Bins, `bounds` and `bin_sizes` are then all integer-reproducible.
- Bin cut: `[y1 + i*h//k, y1 + (i+1)*h//k)`, with row-major channel binding `i*k + j`.
- Empty bins: when `h < k` the leading row bins degenerate to empty spans (a 1×1 RoI with k=2
  leaves pixels only in the last bin); empty bins count as `0.0` and still occupy the `k²`
  denominator — a direct consequence of equal voting and the place where "weight by area" is
  most tempting.
- Zero-area RoIs (`x2 <= x1` or `y2 <= y1`) and fully out-of-bounds RoIs have every bin empty and
  score 0.

## Mutations

All 15 mutants are CAUGHT:

```
m-bin-start / m-bin-end      span start/end formula broken → bin_sizes + values
m-channel-col-major          column-major channel binding   → constant-channel checks
m-empty-ones                 empty bin filled with 1       → empty-bin checks
m-bin-sum                    pooling by sum                → mean checks
m-pooled-sum                 vote by sum                   → pooled assertions
m-pooled-size-weight         vote weighted by area         → 1-pixel vs 3-pixel check
m-no-clip-x1 / m-clip-x2-1   no clip / upper bound off by 1 → out-of-bounds RoI checks
m-neg-width                  width not clamped at 0        → reversed-coordinate check
m-roi-int-check              accepts non-integer coords    → validation check
m-kk-check / m-k-bool        channel-count and bool gates  → validation (bool has its own case)
m-size-count                 pixel count added not multiplied → bin_sizes
m-bounds-order               bounds stored transposed      → asymmetric RoI check
```

## Overlap check

- `vision-yolov3-anchor-decode`(702), `vision-yolov5-build-targets`(707) and
  `vision-yolov8-tal-assign`(708) cover positive assignment and box encoding, not in-RoI bucketing.
- `vision-kl-var-voting`(705) fuses boxes by weight and has no position-sensitive channels.
- `vision-spp-pyramid-pool`(703) is multi-scale pyramid pooling whose bins come from `ceil(H/k)`
  with no channel grouping; the channel-position binding and equal voting here are distinct.
