# Quantization: BAQ Loss-Power Bit Allocation

## Problem

Fixed-bit-width quantization gives every weight column the same number of codewords, but loss (sensitivity) on calibration data varies widely across channels: spending more bits on sensitive channels and fewer on insensitive ones lowers error at the same average bit width. BAQ (arXiv:2506.05664) maps each column's loss to a codeword count with a **power law**, then quantizes the weights asymmetrically.

Given a loss vector `loss` (length equals the column count `C`), a reference bit width `R_ref` (integer) and a mean scaling factor `ratio` (positive), with the fixed exponent `th = 1.95`:

\[
\bar{\ell} = \text{ratio}\cdot\text{mean}(\text{loss}),\qquad
t_c = \max\!\left(\frac{\text{loss}_c}{\bar{\ell}}\cdot 2^{th\cdot R_{ref}},\ 1\right)
\]

\[
\text{levels}_c = \text{round}\!\left(t_c^{1/th}\right),\qquad
\text{maxq}_c = \text{levels}_c - 1,\qquad
\text{bits}_c = \log_2(\text{levels}_c)
\]

- When `loss_c` equals the mean, `levels_c = 2^R_ref`, so that column gets exactly the reference bit width;
- `levels_c` need not be a power of two (e.g. 5 → `maxq = 4`, `bits ≈ 2.3219`); it is the number of codewords available to that column;
- `max()` guarantees `levels_c >= 1`: columns whose loss is far below the mean degrade to `maxq = 0` (0 bits).

Quantization then proceeds element-wise. The dynamic range is taken per **row** (output channel), while the codeword count is taken per **column**:

\[
xmax_r = \max(\max_j W_{r,j}, 0),\qquad
xmin_r = \min(\min_j W_{r,j}, 0)
\]

An all-zero row uses `xmin = -1, xmax = +1`. Hence

\[
scale_{r,c} = \frac{xmax_r - xmin_r}{maxq_c + 10^{-8}},\qquad
zero_{r,c} = \text{round}\!\left(\frac{-xmin_r}{scale_{r,c}}\right)
\]

\[
q_{r,c} = \text{clamp}\!\left(\text{round}\!\left(\frac{W_{r,c}}{scale_{r,c}}\right) + zero_{r,c},\ 0,\ maxq_c\right),\qquad
\hat{W}_{r,c} = scale_{r,c}\cdot(q_{r,c}-zero_{r,c})
\]

## Task

Implement `solve(W, loss, R_ref, ratio)`:

- `W`: float32 `(O, C)`, non-empty and finite;
- `loss`: float32 `(C,)`, nonnegative and finite, with mean greater than 0;
- `R_ref`: `int` in `1..8`;
- `ratio`: `int`/`float`, finite and greater than 0;
- Return a `dict`:
  - `bits` float32 `(C,)`: per-column bit width `log2(levels)`;
  - `maxq` int64 `(C,)`: per-column maximum code `levels - 1`;
  - `scale` float32 `(O, C)`: element-wise scale;
  - `zero` int64 `(O, C)`: element-wise zero point;
  - `dequant` float32 `(O, C)`: dequantized weights.

**Conventions**: float64 internally, float32 output, tolerance `1e-6`; every `round` is round-half-to-even (`torch.round`). The scale denominator is `maxq + 1e-8` (as upstream, not `maxq + 1`). Raise `ValueError` when `W`/`loss` have illegal shapes or types, `loss` contains negative or non-finite values or has zero mean, `R_ref` is not an integer in `1..8`, or `ratio` is non-finite or non-positive.

## Example

```
W = [[0.8, 0.6, 0.9],
     [0.2, 0.6, 0.3]]
loss = [0.3, 1.0, 1.7], R_ref = 2, ratio = 1.0

loss_mean = 1.0, 2^(th*R_ref) = 2^3.9 = 14.9285
t = [4.4786, 14.9285, 25.3785]
t^(1/1.95) = [2.157, 4.000, 5.251] -> levels = [2, 4, 5]
maxq = [1, 3, 4], bits = [1, 2, log2(5) = 2.3219]

row 0: xmin = 0, xmax = 0.9, diff = 0.9 -> scale = [0.9, 0.3, 0.225]
row 1: xmin = 0, xmax = 0.6, diff = 0.6 -> scale = [0.6, 0.2, 0.15]
zero = [[0, 0, 0], [0, 0, 0]]

q       = [[1, 2, 4], [0, 3, 2]]
dequant = [[0.9, 0.6, 0.9], [0.0, 0.6, 0.3]]
```

## Constraints

- The bit width depends only on the column loss; all rows in a column share one `maxq`. The dynamic range is per row, so `scale` is an `(O, C)` element-wise matrix.
- `t` must be clamped to `>= 1` before taking the `th`-th root; `levels` must never reach 0.
- A column with `maxq = 0` has all-zero codes, so its dequantized values are all 0 (this happens when `ratio` is large or the loss is tiny).
- All rounding (`levels`, `zero`, `q`) is half-to-even; upstream's GPTQ error compensation and the `q > 1e3` fallback branch are out of scope.
