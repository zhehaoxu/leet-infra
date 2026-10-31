# Quantization: BiLLM Structural Split and Residual Binarization — Algorithm Walkthrough

## 1. Where this problem comes from

BiLLM (arXiv:2402.10685, `Aaronhuang-778/BiLLM`) splits 1-bit weight quantization into a "structural salient segment plus two 1-bit residual paths". The upstream code behind this problem is:

- `binary.py::high_order_residual`: per-row residual binarization;
- `utils/autosearch.py::structural_searching`: the two-stage search for the salient column count `i` and for the magnitude split;
- `utils/mask.py::generate_structural_mask`: builds `mask1/mask2` from the split value;
- `bigptq.py`: hands the three masks to `Binarization.quantize` and optionally applies GPTQ error compensation.

This problem keeps only the deterministic computation of the first three: no calibration data, no Hessian, no GPTQ compensation, no trained parameters. `up_lim` is the same parameter as `structural_guassian_distribution(..., up_lim=30)` upstream and is supplied by the caller here.

## 2. Algorithm steps

Let `HOR(x, mask, order)` be per-row residual binarization. For each row, using only the columns selected by `mask`:

1. `r = x` (first round);
2. `mu = mean(r over mask)`;
3. `sigma = mean(|r - mu| over mask)`;
4. write back `r_hat = sign(r - mu) * sigma + mu` on those columns;
5. if `order=2`, set `r = x - r_hat`, repeat steps 2–4, and add both round outputs.

Stage 1:

- `count_c = sum_r |W[r, c]|`; sort by `(count, original index)` descending stable and keep the first `up_lim` candidates;
- for `i = 1 .. up_lim-1`, let `mask3` be the first `i` candidates and measure the MSE of `HOR(W, ~mask3, 2) + HOR(W, mask3, 2)` against `W`;
- update only on strict improvement, yielding the best `i`.

Stage 2:

- fix `mask3` and compute `group3 = HOR(W, mask3, 2)`;
- zero the `mask3` columns, flatten `|W|`, and sort ascending;
- take the 81 quantiles `0.10 + 0.01k` (linear interpolation) as candidate splits;
- for each split build `mask2 = (~mask3) & (|W| >= split)` and `mask1 = (~mask3) & (|W| < split)`, and measure the MSE of `HOR(mask1,1) + HOR(mask2,1) + group3`;
- update only on strict improvement; the winning split becomes `split`.

## 3. Source-level details

- **Rows, not columns**: `nanmean(masked_x_tensor, dim=1)` reduces over rows; one row shares `mu/sigma`, so three values in `mask1` on the same row are pulled together toward that row's mean.
- **Masked-out positions**: upstream uses `torch.where(mask, residual, nan)` then `nanmean`. In PyTorch `torch.sign(nan)` returns 0, so `binary * mask` contributes 0 outside the mask and does not poison the sum. This problem implements the same semantics by explicit zeroing instead of relying on `nan` behavior.
- **Empty rows**: when a row has no selected column, `nanmean` is `nan` and upstream replaces it with 0; both order rounds output 0.
- **Quantiles include the zeroed salient columns**: upstream flattens `origin_matrix * (~mask3)`, so salient columns contribute 0 to the sorted vector. With many salient columns the low quantiles are dominated by zeros and `split` can be exactly 0; then `mask1` is empty and every remaining column goes to `mask2`. That is a legal output and must not be "fixed" by raising the quantiles.
- **Boundary membership**: `mask2` uses `>= split` and `mask1` uses `< split`. `split` often equals an actual `|W|` value (interpolation hitting a data point), so the equality rule matters.
- **Two error definitions**: stage 1 compares `group3 + group4` (`group4` is the order-2 reconstruction on `~mask3`); stage 2 compares `group1 + group2 + group3`. The stage-1 `group4` is only a search aid and never appears in the output.

## 4. Determinism boundaries

- **Ties**: `torch.topk` does not guarantee a stable order for equal column norms, so this problem fixes ascending original index; both searches use strict improvement, keeping the first candidate. Both rules must be hard-coded or identical inputs can yield different masks.
- **Quantile grid**: upstream `torch.linspace(0.10, 0.90, 81)` defaults to float32, so quantile values carry ~1e-8 noise. This problem uses the float64 grid `0.10 + 0.01k` with linear interpolation; masks agree on any non-knife-edge input. The difference only perturbs the last digits of a split, not the algorithm.
- **Precision**: statistics, errors, and `split` are float64 internally; `reconstructed`/`split` are float32. When building `mask2 = |W| >= split`, compare against the float64 `split`; rounding it to float32 first can flip an element that sits exactly on the boundary.
- **No randomness**: no RNG, no calibration, no optimizer; the same input always yields the same output.

## 5. How this differs from other problems in the collection

- `quant-pbllm-partial-binarize` (order 620): one global magnitude threshold splits binary/high-bit groups, the binary group uses `sign(x - mean)`, and the high-bit group uses per-row asymmetric integers; there is no structural column selection and no multi-order residual.
- This problem: a **column-norm structural search** chooses the 2-bit salient segment, then a **magnitude two-way split plus order-1 residual** handles the rest, and the salient segment itself uses an **order-2 residual**; three masks and a full reconstruction are returned.
- Codebook problems such as `quant-squeezellm-sensitive-kmeans` and `quant-nf4-quantile-grid` do not overlap: this problem creates no codebook and performs no quantile quantization, only splitting and sign reconstruction.
- The 1-bit sparse near-zero group (upstream `method="prune"`) and GPTQ compensation are out of scope; GPTQ-style compensation is covered by the dedicated GPTQ problems.

## 6. Validation

- `check_example`: hand-computed 2×6, `up_lim=3` example, checking salient columns, `split`, all three masks, and the reconstruction.
- `check_matches_oracle`: five shape/`up_lim` combinations (including `up_lim=2` and `up_lim=number of columns`) against a pure-Python oracle.
- `check_salient_columns_by_norm`: salient columns must be the largest by column absolute sum; with `up_lim=2` only the top column may be chosen, and the output is ascending.
- `check_masks_partition`: the three masks are disjoint and cover everything, `mask3` is whole columns, and `mask1/mask2` are non-empty.
- `check_split_is_masked_quantile`: `split` must be one of the 81 quantiles of `|W|` with salient columns zeroed.
- `check_ties_stable`: when all column norms are equal, candidates follow ascending original index and salient columns are a prefix of `[0, 1, ...]`.
- `check_boundary_up_lim`: `up_lim=2` selects one column, `up_lim=C` selects at most `C-1`, and a single-row matrix works.
- `check_validation`: bad dimensions, empty tensor, non-float, non-finite values, and non-integer or out-of-range `up_lim` raise `ValueError`.
