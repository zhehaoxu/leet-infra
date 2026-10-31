# Notes: SAM Stability Scores and Single-Mask Selection

## Approach

One elementwise comparison per bound feeds the whole computation: compare `masks` against the lower bound `thr - offset`, the upper bound `thr + offset` and the delivery bound `thr`, then collapse each boolean tensor with `flatten(2).sum(dim=-1)` into `(B, C)` integer counts. The summation order matches upstream - row sums first, then the total - but every accumulator stays int64, with no `int16` truncation in between. `intersection <= union` follows from set inclusion, so no clamping is needed before the division.

Handle the zero denominator before dividing: build the `unions == 0` mask, replace the divisor by 1 through `torch.where` (never mutating the original), then overwrite exactly the empty-union positions with `0.0`. That avoids the `nan` of `0/0` and also avoids the dirty trick of adding an epsilon, which would distort the ratio. Cast the integer counts to the masks' dtype before dividing, otherwise float32 inputs get silently promoted to float64.

Build the bias the way upstream does, as a `(1, C)` constant row `[1000, 0, 0, ...]`, multiply it by `(num_points - 2.5)` and add it to `stability` to get `biased`; do not slice-assign into `stability`, which is also returned verbatim. `torch.argmax(biased, dim=1)` yields the smallest index on ties for free. Gather with fancy indexing - `arange(B)` paired with `best_idx` - to obtain `selected_mask`, `selected_score` and `pred_iou_selected` in one pass, already shaped `(B, H, W)`, `(B,)` and `(B,)`.

`areas` is just the count against `thr`, the same bound the selection uses, so `areas[b, best_idx[b]]` necessarily equals the true-pixel count of `selected_mask[b]`. There is no sort, no attention, and no second pass over the pixels.

## Pitfalls and boundaries

- All three bounds are strictly greater-than: a pixel equal to `thr + offset` stays out of the intersection but still enters the union as long as it exceeds `thr - offset`. Writing `>=` instead sweeps the whole zero-valued background into the union and systematically depresses every score.
- `union == 0` must be recorded as `0.0`. Upstream yields `nan` there, and an `argmax` over `nan` is unreliable, so this is a declared deviation rather than an accident.
- The bias applies only to index 0: adding it to every channel is the same as not adding it, and the selection degenerates into a pure-`stability` argmax; adding it to the wrong channel (say the last) makes every documented number miss.
- Dropping the `-2.5` in `(num_points - 2.5)`, or dividing the term by `C`, moves the flip point away from "two points dock 500, three points reward 500".
- With `C == 1` the bias still shows up in `biased` even though the argmax has nothing to choose: `biased` is a returned ledger and must not be skipped just because no comparison happens.
- Ties take the smallest index. Sorting and taking the last element, or using `topk`, can return a larger index for equal values. Sample 1 has no tie (`0.333333` versus `0.666667`), but the seeded sweeps are full of equal ratios from different pixel counts, such as `1/2` and `2/4`.
- Letting `iou_preds` drive the selection is the *other* upstream path, taken only when the stability-score switch is off; here it is a passive ledger.
- An all-false `selected_mask` is legal - the winning channel can have stability 0 - so do not treat it as a degenerate input.
- Shapes and dtypes: `masks` must be 4-D with all dimensions positive, `iou_preds` must be `(B, C)` sharing the masks' dtype and device, and NaN/Inf must raise `ValueError`. `num_points` must be a positive integer and `bool` does not count; `offset` must be a positive real. Integer outputs are `int64`, the boolean output is `bool`, and float outputs reuse the input float dtype.
- Inputs are never mutated and need not be contiguous; do not `contiguous()`-copy a full-size mask stack before `flatten`.

## Independent validation

The oracle runs entirely over Python lists: `masks.tolist()` gives nested lists, four `for` loops count pixels with three plain `if value > bound` tests, the score divides two Python integers, an empty union is written as `0.0` explicitly, the bias is `score + bias if channel == 0 else score`, and the maximum index comes from a hand-written left-to-right scan with a strict `>`; the boolean grids are rebuilt cell by cell with `append(True/False)` and only wrapped into tensors at the very end. Nothing on that path uses torch broadcasting, torch sums or torch `argmax` - only the formulas are shared with the implementation.

Every value in the examples is a dyadic rational (an integer, a quarter or a sixteenth), representable exactly in both float32 and float64, so the Python bound comparisons agree with the torch ones bit for bit and no tolerance is needed to absorb float error.

The 15 `check_*` functions - `check_example_penalty_hides_best_channel` and `check_example_bias_prefers_first_token` are the public examples - cover the two examples, the bias flip as `num_points` runs from 1 to 12, first-index tie-breaking, the three strict bounds, all-foreground and empty-union masks, `areas` tracking `mask_threshold`, `mask_threshold` and `offset` as free parameters with their monotonicity, per-sample row independence and row flipping, `iou_preds` as a passive ledger, minimal shapes with a single candidate, both dtypes, noncontiguous inputs with no mutation, a seeded sweep of 24 shape and threshold combinations, and every `ValueError` documented in the contract. Besides comparing all seven returned tensors, `_verify` independently recomputes shapes and dtypes, asserts scores lie in `[0, 1]`, that `biased[:, 0]` is exactly `stability[:, 0]` plus the bias while other channels are unchanged, that `selected_score` and `pred_iou_selected` are both read at `best_idx`, and that `areas[b, best_idx[b]]` equals the count of `selected_mask[b]`; it additionally requires every channel before `best_idx` to be strictly smaller and every channel after it to be no larger, which pins down "first maximum".

## Complexity

Let `N = B * C * H * W`. The three elementwise comparisons with their collapsed sums, the `(B, C)` divisions, and the bias plus argmax over `(B, C)` are each linear, for a total of `O(N + B * C)`, i.e. `O(N)` since `H * W >= 1`. Peak extra space is the three integer count tensors and the `(B, C, H, W)` boolean keep-mask, i.e. `O(N)`, with the returned data adding `O(B * H * W + B * C)`. Against the mask decoder's attention and upsampling this stage is a pure linear scan with no sorting at all.

## Source and intentional simplifications

Based on [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) (Kirillov et al., arXiv:2304.02643): `segment_anything/utils/amg.py::calculate_stability_score` and `segment_anything/utils/onnx.py::SamOnnxModel.select_masks`. The retained facts are the intersection at `thr + offset` and the union at `thr - offset` with both comparisons strict, the two-level "row sums then total" counting, the resulting `[0, 1]` range, the defaults `Sam.mask_threshold = 0.0` and `stability_score_offset = 1.0`, the `(1, C)` weight row `[1000, 0, ...]`, the `(num_points - 2.5)` multiplier applied only to index 0, `torch.argmax` returning the smallest index on ties, the `arange(B)`-with-`best_idx` gathering of masks and scores, and the delivered mask still being the raw `masks > mask_threshold` binarisation.

Declared deviations and deliberate exclusions:
- Upstream's `SamOnnxModel.forward` carries a `use_stability_score` switch and selects on the raw IoU predictions when it is off. That switch is removed here and the stability score is **always** used, so `iou_preds` degrades into a ledger output; this is our stated simplification, not upstream behaviour.
- Upstream returns `0/0 = nan` when the union is empty; this exercise defines that case as `0.0`, also a declared deviation.
- Upstream's intermediate sums in `calculate_stability_score` use `dtype=torch.int16` and overflow above 32767 pixels before widening to `int32`; here the counting is exact int64 and that quirk is deliberately not reproduced.
- Upstream's ONNX `select_masks` additionally takes `prev_masks` for overlap clipping; this signature has no such input and selection looks only at scores.
- Upstream computes `areas` on the **upscaled** masks and only when `return_extra_metrics` is on; here it is always returned and counted on the raw `H x W` masks.
- Out of scope: the image/pixel resizing itself (PIL/torchvision resampling), `mask_postprocessing` bilinear interpolation and cropping, the `mask_downscaling` Conv2d mask-input path, the ViT image encoder and every mask-decoder transformer/MLP weight, and the hyper-network grid-of-points mask generation - the candidate count is exactly the input `C`, with no per-prompt batch of 32 masks to choose among. The gaussian positional-encoding matrix and the embedding vectors that the stability score never touches belong to the companion problem `mm-sam-point-prompt-embedding`, where they become explicit inputs because upstream initialises them with an unseeded `torch.randn`.
