# FP4 E2M1 Grid Quantization with Two-Level Block Scaling

FP4 is the native inference format of the Blackwell generation: `E2M1` has 1 sign bit,
2 exponent bits and 1 mantissa bit, so the entire positive half-axis has only 8 codes.
It has no `inf`, no `NaN`, and no second non-zero subnormal — and those absences are
exactly where quantization ledgers go wrong.

## Interface

```python
def solve(values, block_size, block_scales, global_scale=1.0) -> dict
```

`values` is a 1-D `float64` tensor; `block_size` is the number of elements per block
(32 for MXFP4, 16 for NVFP4); `block_scales` holds one scalar per block and its length
must equal `ceil(n / block_size)`; `global_scale` is the NVFP4 second level (per tensor).
The returned ledger is:

| Field | Meaning |
| --- | --- |
| `codes` | 4-bit code per element (bit3 sign / bit2..1 exponent / bit0 mantissa) |
| `dequant` | dequantized value (the sign of `-0.0` is preserved) |
| `eff_scale` | per-element effective scale `global_scale * block_scales[i // block_size]` |
| `sat_mask` / `tie_mask` | whether the element saturated / sits exactly on a midpoint |
| `block_sat` / `block_ties` | per-block saturation and tie counts |
| `grid` | the 8 codes on the positive half-axis |

## Input / Output

```
Input: values=[0.25, 0.75, 1.25, 2.5, 7.0], block_size=5, block_scales=[1.0]
Output: codes=[0, 2, 2, 4, 7], dequant=[0.0, 1.0, 1.0, 2.0, 6.0],
      tie_mask=[True, True, True, True, False], sat_mask=[False, False, False, False, True]
```

`0.25` is the midpoint of `0` and `0.5`, so the tie goes to the code with mantissa bit 0,
namely `0`; `7.0` exceeds the largest code `6` and saturates to `6`.

## Rules

- **Positive grid** `0, 0.5, 1, 1.5, 2, 3, 4, 6`: `0.5` is the only non-zero subnormal
  (`e=0, m=1` gives `m/2 · 2^0`). **There is no `0.25` rung**, so `0.1` can only land on `0`.
- **Nearest + RTNE**: on an exact midpoint (`0.25 / 0.75 / 1.25 / 1.75 / 2.5 / 3.5 / 5.0`)
  pick the code whose **mantissa bit is 0** — not the even magnitude. The midpoint around
  `1.25` is `1.0` vs `1.5`, and `1.0` has mantissa bit 0; the midpoint around `3.5` is
  `3.0` vs `4.0`, and `4.0` wins.
- **Saturation, not overflow**: any `|scaled| > 6` becomes `±6` and sets `sat_mask`; `6.0`
  itself is **not** saturated. `inf` saturates too; `NaN` has no code and raises `ValueError`.
- **The sign bit is independent of the magnitude**: `-0.0` encodes to `8` (not `0`) and
  dequantizes back to `-0.0`; every negative value with magnitude below `0.25` becomes `-0.0`.
- **Two-level scale**: compute `eff = global_scale * block_scales[b]` (a product, not two
  successive divisions), then `scaled = v / eff`. MXFP4 has `global_scale == 1` (single
  level); NVFP4 is the nested two-level case.
- **Block wall**: `b = i // block_size`; a short tail block keeps its own scale. The tests
  use "same value, different block" cases to prove scales never leak across the wall.
- **Validation**: `block_size` must be a positive integer (`bool` is rejected),
  `block_scales` must match the block count with finite positive entries, `global_scale`
  must be finite and positive, and `values` must not contain `NaN`; empty input is legal
  (zero blocks, empty `block_scales`).

## Hints

- Detect ties with **exact float equality** (all midpoints are binary-exact); do not infer
  them from dequantized values.
- `torch.signbit` / `math.copysign` are required to tell `±0` apart; `v < 0` misses negative zero.
