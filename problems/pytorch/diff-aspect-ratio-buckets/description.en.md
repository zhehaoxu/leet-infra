# Diffusion Training: Aspect-Ratio Buckets and Nearest-Bucket Assignment

## Problem Statement

Diffusion training usually groups images of different sizes into a discrete set of "resolution buckets" and batches within each bucket. Samples in a batch then share one tensor shape with no per-image padding, while the original aspect ratio is preserved so images are not stretched.

`generate_aspect_ratio_buckets` builds buckets from a pixel budget `resolution²` and a set of aspect ratios: for each ratio \(a=w/h\),

\[ h=\sqrt{\text{resolution}^2/a},\qquad w=h\cdot a, \]

then each side is snapped to a multiple of `divisibility` (and floored at `divisibility`), deduplicated in order. `find_nearest_bucket` assigns an image \((h,w)\) to the closest bucket \((b_h,b_w)\) using the cross-multiplied metric

\[ \left|h\cdot b_w-w\cdot b_h\right|, \]

which compares aspect-ratio distance while using only integer multiplications (no division error). Ties go to the bucket that appears **later** in the list.

## Rules

1. **Bucket generation**: by default use the 11 ratios `(1,1), (4,5), (5,4), (2,3), (3,2), (3,4), (4,3), (9,16), (16,9), (1,2), (2,1)` in that order; a repeated \((h,w)\) is kept only at its first occurrence.
2. **Generation from base resolutions**: if `base_resolutions` is given (each item `(h, w)`), its aspect ratios replace the default table; the source converts each to \((w,h)\) before taking \(w/h\), so the input is `(h, w)` but the produced buckets are still `(h, w)`.
3. **Snapping**: `round(x / divisibility) * divisibility`, where Python's `round` is **banker's rounding** (`.5` goes to even); then `max(divisibility, ...)` as a floor.
4. **Assignment**: for each image \((h,w)\), scan the buckets and take the index with the smallest metric; comparing with `<=` means ties go to the later bucket.

Return `(buckets, assignments)`: the former is an `(B, 2)` int64 tensor (each row `(h, w)`), the latter an `(N,)` int64 tensor of bucket indices.

## Example

`resolution = 512`, `divisibility = 16`, default ratios give 11 buckets:

```
[(512, 512), (576, 464), (464, 576), (624, 416), (416, 624), (592, 448),
 (448, 592), (688, 384), (384, 688), (720, 368), (368, 720)]
```

For `image_sizes = [[512, 512], [768, 512], [512, 768], [1088, 976], [720, 1280]]`:

```
assignments = [0, 3, 4, 1, 8]
```

`1088×976` has metric `57344` against both `(512, 512)` and `(576, 464)` — an exact tie — and the `<=` rule sends it to the later bucket (index 1). Lowering the height slightly to `1070` makes the `1:1` bucket strictly closer, so the assignment goes back to index 0.

## Constraints

- `resolution` and `divisibility` must be positive integers (booleans do not count).
- `image_sizes` must be an `(N, 2)` `int32` / `int64` tensor with positive height and width; `N = 0` is allowed.
- `base_resolutions` is optional; when given it must be a non-empty list/tuple of two-positive-integer `(h, w)` pairs.
- Raise `ValueError` when any of the above fails.
- Every bucket side is a multiple of `divisibility` and at least `divisibility`; with `divisibility=16` each default bucket's pixel count is within about 2% of `resolution²` (coarser granularity or lower resolution drifts further, so "close" is not an equality).
- At low resolution different ratios may snap to the same bucket, so deduplication must preserve order (e.g. `resolution=16`, `divisibility=16` leaves a single `(16, 16)`).
