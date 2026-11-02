# FP6 Dual-Subformat Quantization and 4-in-3 Byte Packing

A 6-bit float does not fit in a byte: `FP6_E3M2` spends 3 bits on the exponent for dynamic
range, `FP6_E2M3` spends 3 bits on the mantissa for precision, and the two subformats are
chosen by tensor range. Storage is the other half of the problem — four 6-bit codes are
exactly 24 bits, so they must be spliced across three bytes.

## Interface

```python
def solve(values, fmt="auto") -> dict
```

`values` is a 1-D `float64` tensor; `fmt` is `"auto"`, `"e3m2"` or `"e2m3"`. The ledger is:

| Field | Meaning |
| --- | --- |
| `fmt` | the subformat actually used (`auto` is resolved to a concrete one) |
| `codes` | 6-bit code per element (bit5 sign / 3- or 2-bit exponent / low mantissa) |
| `packed` | three bytes per four codes (`int64`, values 0–255) |
| `unpacked` | codes recovered from `packed` (must equal `codes`) |
| `dequant` | dequantized values (`-0.0` keeps its sign) |
| `sat_mask` / `tie_mask` | whether the element saturated / sits exactly on a midpoint |
| `pad_count` | how many zero codes were appended to reach a multiple of four |

## Input / Output

```
Input: values=[0.0, 0.25, 0.5, 0.75], fmt="e3m2"
Output: codes=[0, 4, 8, 10], packed=[0, 129, 40], pad_count=0
```

Byte by byte: `0x00` carries the low 6 bits of `c0` plus the low 2 bits of `c1`, `0x81`
carries the high 4 bits of `c1` plus the low 4 bits of `c2`, and `0x28` carries the high
2 bits of `c2` plus all of `c3`.

## Rules

- **Two subformats** (both have no `inf` / `NaN`; the largest value is the saturation point):
  - `e3m2` = 1-3-2, `bias=3`, `max = 1.75 × 2^4 = 28`; smallest subnormal `0.0625`, top
    subnormal `0.1875`, smallest normal `0.25`.
  - `e2m3` = 1-2-3, `bias=1`, `max = 1.875 × 2^2 = 7.5`; smallest subnormal `0.125`, top
    subnormal `0.875`, smallest normal `1.0`.
- **`auto` selection**: `amax <= 7.5` picks `e2m3` (precision first), `amax <= 28` picks
  `e3m2` (range first), anything above `28` raises `ValueError`. An explicit format skips
  the range check and simply saturates.
- **Encoding**: nearest neighbour; ties go to the code with **mantissa bit 0**; any `|x|`
  beyond the format maximum saturates to the max code (all-ones exponent and mantissa) and
  sets `sat_mask`, keeping the sign bit.
- **Packing (4-in-3)**: four 6-bit codes form three bytes as a little-endian bit stream —
  `b0 = c0 | (c1 & 3) << 6`, `b1 = (c1 >> 2) | (c2 & 0xF) << 4`, `b2 = (c2 >> 4) | c3 << 2`.
  Equivalently: read the three bytes as a little-endian 24-bit integer and take
  `bits >> (6k) & 0x3F` for the `k`-th code.
- **Padding**: when the element count is not a multiple of four, append zero codes (value 0)
  with `pad_count = (-n) % 4`; `unpacked` must be truncated back to the original length.
- **Validation**: an illegal `fmt` or a `NaN` in `values` raises `ValueError`; empty input is
  legal (`packed` empty, `pad_count == 0`).

## Hints

- The 6-bit codes cross byte boundaries, so every byte in `packed` must land in `0–255`;
  writing `c1 << 6` without masking first overflows.
- The exponent width differs (3 versus 2), so both the shift amount and the mask in the
  decode formula must follow `fmt`; do not mix the two layouts.
