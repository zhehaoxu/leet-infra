# Quantization: AFPQ Dual-Scale FP4 Quantization

## Problem

FP4/FP3 codebooks are not evenly spaced: grid points are dense near small values and sparse near large ones — the advantage of a floating-point representation. But if the positive and negative sides share one scale, an asymmetric weight distribution wastes codes on the small side.

AFPQ (Asymmetric Floating Point Quantization for LLMs, 2023, arXiv:2311.01792) gives **each side its own scale**:

- `scale_pos = xmax / (maxq/2)`, determined only by the positive extreme;
- `scale_neg = |xmin| / (maxq/2)`, determined only by the negative extreme.

The positive and negative parts are divided by their own scales, mapped to the nearest codebook entry, and reconstructed and summed. When the two extremes differ greatly, the smaller side still lands on dense code points.

This problem uses the two codebooks from the source (`maxq=12` for `bits=4`, `maxq=8` for `bits=3`):

- 4-bit: `[-6, -4, -3, -2, -1.5, -1, -0.5, -0, 0, 0.5, 1, 1.5, 2, 3, 4, 6]`
- 3-bit: `[-4, -2, -1, -0, 0, 1, 2, 4]`

The table contains both `-0` and `0`; they are numerically equal. When two codes tie for nearest, take the one **earlier** in the table (the first `argmin` match).

## Task

Implement `solve(W, bits)`:

- `W`: float32 `(G, N)`, quantized independently per row (output channel);
- `bits`: either `3` or `4`.
- Per row:
  1. `xmin = min(row.min(), 0)`, `xmax = max(row.max(), 0)`; if both are 0, set `xmin=-1`, `xmax=+1`;
  2. `scale_pos = xmax / (maxq/2)`, `scale_neg = |xmin| / (maxq/2)`;
  3. Split each element into `x_pos = max(w, 0)` and `x_neg = min(w, 0)`, divide by `scale_pos/scale_neg`, and take the nearest code to get `q_pos/q_neg`;
  4. `dequant = q_pos * scale_pos + q_neg * scale_neg`.
- Return a `dict`: `scale_pos`, `scale_neg` float32 `(G,)`; `q_pos`, `q_neg`, `dequant` float32 `(G, N)`.

**Conventions**: float64 internally, float32 output, tolerance `1e-6`. When a row has no positive (or no negative) values, that side has `scale=0` and both its code and reconstruction contribution are 0 — **no 0/0 division**. A non-3/4 `bits`, a non-2-D or non-floating `W`, an empty tensor, or non-finite values raise `ValueError`.

## Example

```
W = [[3.0, -0.3, 0.2]], bits = 4, maxq = 12, half = 6
xmax=3.0, xmin=-0.3 -> scale_pos=0.5, scale_neg=0.05
q_pos = [6, 0, 0.5]     (0.2/0.5=0.4, nearest code 0.5)
q_neg = [0, -6, 0]      (-0.3/0.05=-6, exactly on a code point)
dequant = [3.0, -0.3, 0.25]
```

Compare with a single scale: sharing `3.0/6=0.5` for both sides maps `-0.3/0.5=-0.6` to `-0.5` only, reconstructing `-0.25` with error `0.05`; the dual scale represents `-0.3` exactly.

## Constraints

- Codebooks must match the tables exactly; `q_pos/q_neg` may only take table values.
- Ties pick the earlier code in the table, not the later one.
- Rows with one missing side (all-positive/all-negative) contribute zero from that side; all-zero rows degenerate to `(-1, +1)`.
- Compute in float64 so a float32 division cannot flip the nearest code at a boundary.
