# notes (vision-dfl-distribution-box)

## Source

- Loss: `open-mmlab/mmdetection` main, `mmdet/models/losses/gfocal_loss.py::distribution_focal_loss`
  (lines 144-165). The upstream body is `dis_left = label.long()`, `dis_right = dis_left + 1`,
  `weight_left = dis_right.float() - label`, `weight_right = label - dis_left.float()`,
  `loss = CE(pred, dis_left) * weight_left + CE(pred, dis_right) * weight_right`.
- Decoding: the GFL head's `integral` — `softmax(x.reshape(-1, reg_max+1))` combined linearly with
  `arange(reg_max+1)`; `distance2bbox` then multiplies by the stride, which is an explicit input here.
- Paper: arXiv 2006.04388 (Generalized Focal Loss) §3.2.

## Traps in the upstream implementation

1. **Two CEs, not one**: `distribution_focal_loss` computes `cross_entropy` at `left` and at
   `right` separately and weights them by `wl`/`wr`; treating the target as a soft label in a
   single CE gives different gradients and numbers.
2. **The weights sum to 1**: `wl = right - label`, `wr = label - left`, and `right = left + 1`,
   so `wl + wr = 1`. Writing `wl = label - left` happens to still give 0/1 on integer targets, so
   only a fractional target exposes the swap.
3. **`label.long()` truncates toward zero**: equivalent to floor for positive targets, but it is
   not `round` — `3.7` must land in bin 3, not 4.
4. **`right` can go out of bounds**: with `label = reg_max`, `right = reg_max + 1` exceeds the bin
   range and the upstream code crashes; this problem normalizes it to `ValueError` and requires
   `target ∈ [0, reg_max-1]`.
5. **The expectation is 0-based**: `arange(reg_max+1)` starts at 0, so a uniform distribution has
   expectation `reg_max/2`, not `(reg_max+1)/2`.
6. **The normalizer**: `@weighted_loss` with `reduction='mean'` divides by the element count
   (`N × 4`), not by the number of positives.

## Normalizations in this problem

- The side count `S` is not hard-wired to 4 (its detection value) so single-side cases can isolate
  the weights, indices and CEs.
- The ledger exposes `left`/`right`/`w_left`/`w_right`, making "two CEs" and "weight direction"
  observable instead of inferring them from a scalar loss.
- Decoding and loss are decoupled: without `targets` only `probs`/`dist`/`bin_index` are returned
  and no loss key may appear.
- Probabilities use `softmax`; the loss uses `log_softmax` + gather, matching `F.cross_entropy`.
- Validation gates: `reg_max` positive integer (`bool` rejected with its own case so the shape
  check cannot mask it), last dimension exactly `reg_max+1`, `stride` finite and positive, and
  `targets`/`weights` shapes matching `dist`.

## Mutations

All 14 mutants are CAUGHT:

```
m-bin-offset                 indices start at 1            → uniform-expectation check
m-dist-no-stride             decoding drops the stride      → stride check
m-wl-swap                    left/right weights swapped     → fractional-target check
m-single-ce / m-ce-no-weight one CE / weights not applied   → mixture checks
m-right-eq-left              right equals left              → index assertions
m-target-gate                out-of-range target allowed    → range check
m-mean-denom                 denominator uses N             → normalization check
m-no-weights                 weights ignored                → weight check
m-softmax-dim                softmax on the wrong dim       → decode check
m-left-round                 truncation replaced by round   → 3.7-style check
m-regmax-bool / m-logit-dim  two validation gates broken    → validation (bool has its own case)
m-stride-finite              finiteness check dropped       → stride=inf check
```

## Overlap check

- `vision-yolov3-anchor-decode`(702) / `vision-yolov5-build-targets`(707) cover box encoding and
  target assignment with no distributional regression.
- `vision-yolox-simota-assign`(706) is about assignment cost, not DFL.
- The only subject here is "distributional expectation decoding plus the two-CE mixture".
