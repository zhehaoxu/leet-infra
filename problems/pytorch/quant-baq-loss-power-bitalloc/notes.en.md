# Quantization: BAQ Loss-Power Bit Allocation — Algorithm Notes

## 1. Why allocate bits by a power of the loss

BAQ (arXiv:2506.05664) observes that quantization error affects each input channel (column) differently, while fixed bit widths spread codewords evenly across all columns. It uses the calibrated loss `loss_c` of each column as importance and builds a curve mapping **relative loss** to **codeword count**. This problem follows upstream's power-law form with the fixed exponent `th = 1.95`:

\[
\text{levels}_c = \text{round}\!\left(\left(\max\!\left(\frac{loss_c}{\bar\ell}\cdot 2^{th R_{ref}},1\right)\right)^{1/th}\right)
\]

The key is the normalizer `2^{th R_ref}`: when `loss_c = \bar\ell` the bracket is exactly `2^{th R_ref}`, whose `th`-th root is `2^{R_ref}`, so `levels = 2^{R_ref}` and `maxq = 2^{R_ref}-1` — **the mean-loss column receives exactly the reference bit width**. Every time the loss drops to `2^{-th}` of its value, the codeword count halves and the bit width drops by 1; the exponent `th > 1` flattens the curve near the mean so small loss fluctuations do not add or remove a bit.

## 2. Bit width is continuous, not an integer

`levels_c` is a **codeword count** after rounding and can be any positive integer: `levels = 5` gives `maxq = 4` and `bits = log2(5) ≈ 2.3219`. The `bits` output is therefore a float describing how many bits the column actually uses; what enters the quantizer is `maxq`, the largest code, and `[0, maxq]` holds `maxq + 1 = levels` codewords.

`t_c` is clamped with `max(·, 1)` before the root, guaranteeing `levels >= 1` — the 0-bit lower bound. Columns whose loss is far below the mean (or with a large `ratio`) land on `levels = 1, maxq = 0`, where `q` can only be 0 and the whole column dequantizes to 0.

## 3. Dynamic range per row, codeword count per column

This is the easy-to-misread part of the BAQ source: `fasterquant_baq` takes extremes with `W.max(dim=1)` / `W.min(dim=1)` — per **row** — while `R_vec` and `maxq_scalar` are per **column**. Hence

\[
scale_{r,c} = \frac{xmax_r-xmin_r}{maxq_c + 10^{-8}},
\qquad
zero_{r,c} = \text{round}\!\left(\frac{-xmin_r}{scale_{r,c}}\right)
\]

`xmax`/`xmin` are both clamped toward 0 so the quantization interval covers 0 (important for mostly nonnegative post-ReLU weights). For an all-zero row upstream substitutes `xmin/xmax = -1/+1` to avoid `diff = 0`; the later `diff == 0 → +1e-8` fallback cannot trigger in this flow. `zero = round(-xmin/scale)` maps `xmin` to code 0, and `q` is clamped to `[0, maxq_c]` — because each column has a different codeword count, codes from different columns of the same `scale` matrix are not comparable.

The denominator is `maxq + 1e-8`, not `maxq + 1`, exactly as upstream: it places `xmax` near code `maxq` (rather than `maxq+1`), so the top step misses `xmax` by about 1e-8 relative, which is negligible.

## 4. Boundary behavior

- `maxq = 0`: all codes are 0, and the dequantized value is `scale · (0 - zero)`. Usually `scale` is huge and `zero ≈ 0`, so the result is all zeros.
- Scaling `ratio` up multiplies `loss_mean`, shrinking every `t_c` by `ratio^{-1/th}` and lowering all bit widths proportionally; halving `ratio` reverses it.
- All-zero row: `xmin=-1, xmax=+1`, `diff=2`, but zero weights still quantize to the code equal to `zero` and reconstruct exactly 0.
- Upstream `quantize_vec` rewrites `q > 1e3` using `x_mean`; that branch patches GPTQ's per-column compensation path. This problem does not implement error compensation and therefore not that branch either.

## 5. Division of labor within the collection

- `gptq-column-quantize`: GPTQ per-column quantization with Hessian error compensation, fixed bit width;
- `quant-awq-clip-scale-search`: searches a clipping factor to minimize output error;
- `quant-squeezellm-sensitive-kmeans`: clumps sensitive channels with clustering and keeps them at high precision, still centered on fixed codeword positions;
- `quant-rptq-channel-reorder-group`: channel reordering plus grouping, with a single bit width per group;
- This problem: **allocates a codeword count per input channel** (variable width, possibly 0), then quantizes asymmetric per-row dynamic ranges — allocation and quantization only, without calibration forward passes, Hessian compensation, or the fallback branch.

## 6. Validation

- `check_example`: hand-computed 2×3 case checking `maxq`, `zero`, `bits`, `scale` and `dequant` field by field.
- `check_allocation`: bit widths follow the power law exactly and are monotone nondecreasing in loss; a column at the mean loss takes the reference width `R_ref`.
- `check_row_stats`: `scale[r,c]` equals the row's dynamic range divided by that column's `maxq + 1e-8`, and `zero` follows the row's `xmin`.
- `check_matches_oracle`: three random inputs at different scales match a pure-Python oracle on every field.
- `check_quant_grid`: dequantized values lie on the integer grid of `scale`, with codes integral and within `[0, maxq]`.
- `check_boundary_allocation`: a tiny loss degrades to 0 bits with an all-zero column; a large loss earns more codewords; scaling `ratio` by 16 lowers the reference width from 4 to 2.
- `check_zero_rows`: an all-zero row uses `(-1, +1)` range yet reconstructs zero weights as 0.
- `check_validation`: illegal shapes/types, negative/all-zero/non-finite loss, and bad `R_ref`/`ratio` raise `ValueError`.
