# Multimodal: SAM Stability Scores and Single-Mask Selection

## Task

Turn the raw mask logits produced by the mask decoder into the bookkeeping that the last step of the ONNX export path actually consumes: a stability score per mask token, a selection score biased by the number of point prompts, the single binary mask that gets selected, and ledgers for areas and raw IoU predictions.

This is pure elementwise comparison and integer counting: no interpolation, no morphological post-processing, no mask downscaling, and no learnable weights. Both `mask_threshold` and `offset` are supplied by the caller here so that the threshold convention can be tested directly.

The bilingual starter contract is the single reference for parameters, shapes, numeric limits, and exceptions. The rules below define the computation.

## Rules

1. Three strict thresholds: for every `(b, c)` mask, the intersection count is the number of pixels with `masks > mask_threshold + offset`, the union count is the number with `masks > mask_threshold - offset`, and the area is the number with `masks > mask_threshold`. All three are strictly greater than, so a pixel sitting exactly on a bound falls into the wider bucket. The contract guarantees `offset > 0`, hence the union set always contains the intersection set.
2. The counts are exact integers: sum over the last two dimensions first, which gives three integers per token. Counting is independent of the floating dtype and no intermediate truncation or overflow is allowed.
3. Stability = intersection count / union count; when the union count is 0 this exercise defines the stability as `0.0` (upstream produces `0/0 = nan` there). Stability therefore always lies in `[0, 1]`, and a `0.0` can come either from an empty union or from an empty intersection with a non-empty union.
4. The bias touches only token 0: `biased[b, c] = stability[b, c] + (num_points - 2.5) * (1000.0 if c == 0 else 0.0)`. With `num_points <= 2` token 0 is pushed down by at least 500, and with `num_points >= 3` it is pushed up by at least 500, so the switch happens between 2 and 3; `num_points` is a positive integer, so `2.5` itself is unreachable and the bias is never zero.
5. Selection looks only at `biased`: `best_idx[b]` is the index of the maximum of `biased[b]`, and on ties the smallest index wins (the behaviour of torch `argmax`, not of a stable sort). `iou_preds` never takes part in the selection.
6. The selected entry: `selected_mask[b] = (masks[b, best_idx[b]] > mask_threshold)` is a boolean mask, and `selected_score[b] = stability[b, best_idx[b]]` is the unbiased stability score, not `biased`.
7. The ledgers: `areas[b, c]` records per token the number of pixels with `masks > mask_threshold`, using `mask_threshold` rather than the two stability bounds; `pred_iou_selected[b] = iou_preds[b, best_idx[b]]` merely gathers the raw prediction of the selected token.

Return a seven-tuple: `selected_mask` bool `(B,H,W)`, `selected_score` `(B,)`, `stability` `(B,C)`, `biased` `(B,C)`, `best_idx` int64 `(B,)`, `areas` int64 `(B,C)`, `pred_iou_selected` `(B,)`. Float outputs reuse the dtype of `masks`; integer outputs are int64.

## Example 1: Two points knock token 0 out

All floating tensors are float64, with the defaults `mask_threshold = 0.0` and `offset = 1.0`.

```text
masks = [[[[2.0, 2.0], [2.0, 2.0]],
          [[1.0, 0.0], [-1.0, 2.0]],
          [[0.5, 2.0], [2.0, -2.0]]]]
iou_preds = [[0.9375, 0.5, 0.875]]
num_points = 2
mask_threshold = 0.0
offset = 1.0

stability = [[1.000000, 0.333333, 0.666667]]
biased = [[-499.000000, 0.333333, 0.666667]]
best_idx = [2]
selected_score = [0.666667]
selected_mask = [[[True, True], [True, False]]]
areas = [[4, 2, 3]]
pred_iou_selected = [0.875000]
```

The intersection/union counts of the three tokens are `4/4`, `1/3` and `2/3`. Token 0 has the highest stability yet the bias `(2 - 2.5) * 1000 = -500` drags it down to `-499`, so index 2 is selected. In the mask at index 1, `1.0` equals `mask_threshold + offset`, `0.0` equals `mask_threshold` and `-1.0` equals `mask_threshold - offset`: all three bounds exclude their exact matches because the comparison is strictly greater than, which is why the `-1.0` pixel stays out of the union. The selected binary mask uses `mask_threshold`, and it agrees with `areas[0, 2] = 3` pixel for pixel.

## Example 2: Three points and an empty union

All floating tensors are float64, with `mask_threshold = 0.0`, `offset = 1.0` and `num_points = 3`.

```text
masks = [[[[1.5, -0.5], [-1.5, 0.5]],
          [[-1.0, -2.0], [-3.0, -1.5]],
          [[2.0, 2.0], [2.0, 2.0]]],
         [[[-2.0, -2.0], [-2.0, -2.0]],
          [[1.0, 1.0], [1.0, 1.0]],
          [[0.5, -0.5], [2.0, 3.0]]]]
iou_preds = [[0.25, 0.5, 0.75], [1.0, 0.0625, 0.5]]
num_points = 3
mask_threshold = 0.0
offset = 1.0

stability = [[0.333333, 0.000000, 1.000000], [0.000000, 0.000000, 0.500000]]
biased = [[500.333333, 0.000000, 1.000000], [500.000000, 0.000000, 0.500000]]
best_idx = [0, 0]
selected_score = [0.333333, 0.000000]
selected_mask = [[[True, False], [False, True]],
                 [[False, False], [False, False]]]
areas = [[2, 0, 4], [0, 4, 3]]
pred_iou_selected = [0.250000, 1.000000]
```

The `+500` bias makes token 0 win in both batches, even in the second batch where its stability is `0.0`: there all four pixels are `<= mask_threshold - offset`, the union count is 0, and this exercise returns the agreed `0.0` instead of `nan`, with an all-`False` selected mask. In that same batch the token at index 1 has all four pixels exactly equal to `mask_threshold + offset`, so the intersection is empty while the union is 4 and the score is `0.0` as well - the empty-intersection and empty-union routes to `0.0` must both be reachable. Batch bookkeeping never interferes: the first batch has area 0 at index 1 while the second batch has area 4 there.

All displayed outputs were dumped from actual reference execution and checked against an independent oracle; values are rounded to six decimal places.
