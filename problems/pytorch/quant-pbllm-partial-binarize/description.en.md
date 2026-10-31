# Quantization: PB-LLM Partial Binarization

## Problem

Compressing every weight to 1–2 bits lets the error on **salient weights** explode: a few large-magnitude weights dominate the output of an LLM, while most small-magnitude weights are insensitive. PB-LLM (Partially Binarized LLMs, 2023, arXiv:2310.00034) binarizes **only the small weights** and keeps the large ones at a higher bit width:

1. Select the smallest-magnitude elements by `|W|` and mark them in `mask`;
2. Elements inside `mask` go through a 1-bit binary (xnor) path;
3. All remaining elements go through `high_bits`-bit asymmetric quantization;
4. Reconstructed values from the two paths are selected elementwise by `mask`.

This problem fixes `groupsize=-1` (the whole weight tensor is one low group) and implements only the RTN branch, without GPTQ's Hessian error compensation.

**Mask**: flatten `|W|` globally and sort ascending; the value at index `int(W.numel() * low_frac)` is the threshold `thresh`, and `mask = |W| <= thresh`. Ties enter together, so the actual fraction can be slightly above `low_frac`.

**Low path (xnor)**: first compute `masked = W * mask` (non-selected positions are zero, and **zeros participate in the statistics**), then per row

- `low_mean = masked.mean(dim=1)`
- `low_scale = |masked - low_mean|.mean(dim=1)`

Reconstruction is `sign(w - low_mean) * low_scale + low_mean` (`torch.sign(0) == 0`).

**High path (per-row asymmetric)**: `xmin=min(row.min(), 0)`, `xmax=max(row.max(), 0)`; if both are 0, use `(-1, +1)`; `maxq=2^high_bits-1`; `high_scale=(xmax-xmin)/maxq`; `high_zero=round(-xmin/high_scale)`. Reconstruction:

`q = clamp(round(w / high_scale) + high_zero, 0, maxq)`, `w_hat = (q - high_zero) * high_scale`.

## Task

Implement `solve(W, low_frac, high_bits)`:

- `W`: float32 `(O, C)` weights, non-empty and finite;
- `low_frac`: a number with `0 <= low_frac < 1`, the low-precision fraction;
- `high_bits`: an integer in `2..8`, bit width of the high path.
- Return a `dict`: `mask` bool `(O, C)`; `low_mean`, `low_scale`, `high_scale` float32 `(O,)`; `high_zero` int64 `(O,)`; `dequant` float32 `(O, C)`.

**Conventions**: float64 internally, float32 output, tolerance `1e-6`. Invalid inputs (out-of-range `high_bits`, `low_frac` outside `[0,1)`, non-2-D/non-floating/empty/non-finite `W`) raise `ValueError`.

## Example

```
W = [[1.0, -0.5, 0.25, -0.25, 0.1, -0.1]], low_frac = 0.5, high_bits = 4
sorted |W| = [0.1, 0.1, 0.25, 0.25, 0.5, 1.0], index int(6*0.5)=3 -> thresh = 0.25
mask = [False, False, True, True, True, True]
low: masked = [0, 0, 0.25, -0.25, 0.1, -0.1] -> low_mean = 0, low_scale = 0.7/6
high: xmin=-0.5, xmax=1.0 -> high_scale = 1.5/15 = 0.1, high_zero = round(0.5/0.1) = 5
dequant = [1.0, -0.5, 0.116667, -0.116667, 0.116667, -0.116667]
```

## Constraints

- The mask threshold is taken over the **entire `W`** (`groupsize=-1` in this problem), at sorted index `int(numel * low_frac)`.
- The low `mean/scale` must be based on `W * mask` (non-selected positions participate as zeros), not only on the selected elements.
- `maxq` on the high path is `2^bits - 1` (an all-ones unsigned code); `zero` rounds `-xmin/scale` half-to-even.
- The final reconstruction selects the two paths elementwise by `mask`; it does not add them together.
