# Quantization: AFPQ Dual-Scale FP4 Quantization — Algorithm Notes

## 1. The blind spot of a single scale

FP4/FP3 codebooks are dense near zero and sparse near the extremes, which matches bell-shaped weight distributions. But as soon as both sides share one scale, an asymmetric distribution destroys that advantage. Suppose the positive extreme is 3.0 and the negative extreme is only -0.3. With `scale=3.0/6=0.5`, every negative value collapses around `-0.5`, and `-0.3` reconstructs as `-0.25`.

AFPQ (Asymmetric Floating Point Quantization for LLMs, 2023, arXiv:2311.01792) makes a direct change: each side gets a scale from **its own** extreme. In the upstream implementation, `sym=False` and `two_scale=True`; `find_params` computes `xmin=min(min,0)`, `xmax=max(max,0)`, then divides each by `maxq/2`. `quantize_2scale` splits `x` into positive and negative parts, takes the nearest code for each, and returns `xq = q_pos*scale_pos + q_neg*scale_neg`.

## 2. Codebook and scale correspondence

The largest code is exactly `maxq/2`:

- 4-bit: `maxq=12`, `half=6`, codes `[-6, -4, -3, -2, -1.5, -1, -0.5, -0, 0, 0.5, 1, 1.5, 2, 3, 4, 6]`;
- 3-bit: `maxq=8`, `half=4`, codes `[-4, -2, -1, -0, 0, 1, 2, 4]`.

Hence `scale_pos=xmax/(maxq/2)` maps the positive extreme exactly onto the largest positive code `maxq/2`, and the negative side maps onto the most negative code. Dividing by `maxq/2` rather than another constant keeps the extreme inside the table.

All table values (`±0.5`, `±1.5`, ...) are binary-exact and representable in float32, so the float32 `q_pos/q_neg` outputs match the float64 intermediates bit-for-bit and can be compared exactly.

## 3. Why the split is necessary

Taking the nearest code of `x/scale` directly cannot work: the two sides have different scales. First split into `x_pos=max(x,0)` and `x_neg=min(x,0)`, divide each by its own scale, and map each to a code. Reconstruction adds the two parts; exactly one is zero, so nothing is double-counted.

When a row has no positive values, `xmax` is clamped to 0 by `max(...,0)` and `scale_pos=0`. Upstream hits an undefined `0/0=NaN` path here. This problem specifies that **a side with scale 0 contributes code 0 and reconstruction 0**, avoiding NaN propagation and giving one-sided rows a deterministic result.

## 4. Ties, zeros and degeneracies

- The table contains both `-0` and `0`, numerically equal; `argmin` takes the first match, so the code near zero is `-0` and the reconstruction is still 0.
- Other ties (e.g. `q=1.75` equidistant from `1.5` and `2`) also take the earlier code `1.5`. The first-match rule is part of the contract to prevent implementation drift.
- All-zero row: `xmin=xmax=0`, mapped to `(-1,+1)` as in the source; both scales are `1/(maxq/2)` and the reconstruction is 0.
- `bits ∈ {3,4}`; `W` must be 2-D, non-empty and finite; float64 internally, float32 output.

## 5. Division of labor within the collection

- `quant-nf4-quantile-grid`: NF4 uses a quantile-derived codebook, unlike the fixed FP4/FP3 tables here;
- `symmetric-quantize` / `asymmetric-quantize`: fake-quant on uniform integer grids; this problem uses a non-uniform floating-point codebook;
- `quant-double-quant-scale`: quantizes the scale again; this problem only changes the **per-side** scale choice;
- `gptq-column-quantize`: cross-column error compensation; this problem is row-independent with no compensation;
- This problem: AFPQ's **dual-scale + nearest-code** rounding and its tie rule.

## 6. Validation

- `check_example`: hand-computed 3.0/-0.3/0.2 case covering both scales and codebook rounding.
- `check_codebook`: for 3/4-bit, `q_pos/q_neg` may only take table values and stay within `maxq/2`.
- `check_scales`: `scale_pos/scale_neg` formulas on four row types (asymmetric, all-zero, all-negative, all-positive).
- `check_matches_oracle`: 30 random configurations (including one-sided rows) compared against a pure-Python scalar oracle.
- `check_benefit_vs_single_scale`: with very different extremes, the dual-scale error is far below the single-scale error.
- `check_tie_picks_first_code`: `q=1.75` must pick `1.5`, validating the tie rule.
- `check_one_sided_rows` / `check_all_zero_row`: one-sided rows avoid 0/0; all-zero rows degenerate to `(-1,+1)`.
- `check_validation`: invalid `bits`, wrong dimensionality, empty tensors, non-floating and non-finite inputs raise `ValueError`.
