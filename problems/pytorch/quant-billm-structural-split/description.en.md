# Quantization: BiLLM Structural Split and Residual Binarization

## Problem

The error of 1-bit quantization comes mostly from a few large-magnitude weights being collapsed onto a single sign. BiLLM (arXiv:2402.10685) does not push every weight through one 1-bit path. It first performs a **structural split**: the columns with the largest column norms become a 2-bit "salient" segment, and the remaining weights are split by magnitude into a high and a low 1-bit segment. Inside every segment it uses **per-row residual binarization**: repeatedly subtract the row mean, set a scale from the mean absolute deviation, and take the sign, until the segment's order is exhausted.

This problem implements the weight-only, fully deterministic part: the two-stage split search plus residual-binarized reconstruction. The optional GPTQ error compensation in upstream `bigptq.py`, and the calibration-based Hessian metric, are out of scope.

Given a weight matrix `W` (`O` rows, `C` columns) and a candidate bound `up_lim`:

1. Column salience is `count_c = sum_r |W[r, c]|`. Sort columns by `(count, original index)` in descending stable order and keep the first `up_lim` as candidates;
2. **Stage 1 (structural)**: for `i = 1 .. up_lim-1`, let `mask3` be the first `i` candidates and `~mask3` the rest, residual-binarize both with order=2, and pick the `i` that minimizes the full-matrix MSE of their sum against `W`;
3. **Stage 2 (magnitude split)**: fix the best `mask3`, zero its columns, flatten `|W|` and sort it, then search for a split value `split` over the 81 quantiles `0.10, 0.11, ..., 0.90`:

\[
mask_2 = (\neg mask_3) \wedge (|W| \ge split), \qquad
mask_1 = (\neg mask_3) \wedge \neg mask_2
\]

   Residual-binarize both with order=1, add the stage-1 `group3`, and pick the `split` that minimizes the full-matrix MSE;
4. Return the masks and the reconstruction at the best split.

Residual binarization `HOR(x, mask, order)` is defined per row, over the columns selected by `mask` only:

\[
\mu = \mathrm{mean}_{c \in mask}\, r_c, \qquad
\sigma = \mathrm{mean}_{c \in mask}\, |r_c - \mu|, \qquad
\hat r_c = \mathrm{sign}(r_c - \mu)\,\sigma + \mu
\]

The first round uses `r = x`; for `order=2` the second round replaces `r` by `x - \hat r` and accumulates. Columns outside `mask` output 0, a row with no selected column outputs 0, and `sign(0) = 0`.

## Task

Implement `solve(W, up_lim)`:

- `W`: float32 `(O, C)`, non-empty and finite;
- `up_lim`: `int` with `2 <= up_lim <= C`;
- Return a `dict`:
  - `salient_cols` int64 `(k,)`: columns chosen in stage 1, ascending;
  - `split` float32 scalar: the best stage-2 split;
  - `mask1` / `mask2` / `mask3` bool `(O, C)`: three disjoint masks whose union covers every element;
  - `reconstructed` float32 `(O, C)`: `hor(mask1,1) + hor(mask2,1) + hor(mask3,2)`.

**Conventions**: compute in float64 and output float32; equal column norms keep ascending original index (stable sort); both searches update only on strict improvement, so ties keep the first candidate (smaller `i`, then smaller quantile). Quantiles are `0.10 + 0.01k` (`k = 0..80`, float64) with linear interpolation. Raise `ValueError` when `W` is not a non-empty finite 2-D float tensor, or when `up_lim` is not an integer satisfying the condition above.

## Example

```
W = [[0.5, 3.0, 1.0, 2.0, 0.25, 4.0],
     [0.1, -2.5, 0.5, -1.0, 0.1, -3.5]]
up_lim = 3

column counts (sum |W[:, c]|) = [0.6, 5.5, 1.5, 3.0, 0.35, 7.5]
candidates desc (idx, count)  = [(5, 7.5), (1, 5.5), (3, 3.0)]
i = 1: mask3 = {5},      MSE = 0.0451466
i = 2: mask3 = {5, 1},   MSE = 0.0167025   <- optimal
salient_cols = [1, 5]
split = 0.515

mask3 = [[0, 1, 0, 0, 0, 1],
         [0, 1, 0, 0, 0, 1]]
mask1 = [[1, 0, 0, 0, 1, 0],
         [1, 0, 1, 0, 1, 0]]
mask2 = [[0, 0, 1, 1, 0, 0],
         [0, 0, 0, 1, 0, 0]]
reconstructed = [[0.5,    3.0,  1.0,    2.0,  0.25,   4.0],
                 [0.0556, -2.5, 0.4111, -1.0, 0.0556, -3.5]]
```

In the second row, column 0 belongs to `mask1`; the three `mask1` values `[0.1, 0.5, 0.1]` have mean `0.2333` and mean absolute deviation `0.1778`, so the reconstruction is `0.2333 - 0.1778 = 0.0556`. Column 2 gives `0.4111` the same way.

## Constraints

- **Weight-only**: no calibration forward pass, no Hessian estimate, no trained parameters; `up_lim` only bounds the candidate columns and `i` ranges over `[1, up_lim)`.
- **Stage order matters**: fix the structural `mask3` first, then split the remaining columns by magnitude; splitting by magnitude first gives a different answer.
- **The quantile vector includes zeroed salient columns**: the flattened values are `W * (~mask3)`, where salient columns contribute 0 and pull the low quantiles down. This is upstream behavior; do not compute quantiles on the remaining columns only.
- **Tie rules are part of the spec**: stage 1 updates only on strict improvement (ties keep the smaller `i`), stage 2 likewise (ties keep the smaller quantile), and sorting ties keep ascending original index.
- This problem excludes BiLLM's 1-bit sparse near-zero group (`prune`), GPTQ compensation, and the Hessian-metric branch; it also has no groupsize.
- Difference from `quant-pbllm-partial-binarize` (order 620): PB-LLM uses one global magnitude threshold to split binary/high-bit groups; this problem uses a **column-structured search** for the 2-bit salient group, then a **magnitude two-way split plus multi-order residual binarization** on the rest.
