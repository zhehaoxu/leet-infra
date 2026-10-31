# Sparsification: Wanda Activation-Weighted Magnitude Pruning

## Problem

Weight magnitude alone is misleading: a tiny weight multiplied by a huge input activation still matters, while a medium weight multiplying near-zero activations can be dropped almost for free. Wanda (*A Simple and Effective Pruning Approach for Large Language Models*, 2306.11695) therefore scores each element by **weight magnitude × the RMS of its input channel**:

\[
m_{ij} = |W_{ij}|\cdot \sqrt{\frac{1}{N}\sum_{t=1}^{N} X_{t,j}^{2}}
\]

where \(X\in\mathbb{R}^{N\times d_{\text{in}}}\) is the flattened layer input collected on calibration data. Each **output row** independently drops its smallest-\(m\) columns; rows do not share a threshold and no Hessian is involved. Upstream `lib/prune.py::prune_wanda` writes:

```python
W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
```

`scaler_row` is accumulated in `lib/layerwrapper.py::WrappedGPT.add_batch` as `norm(inp, p=2, dim=1) ** 2 / nsamples`; omitting the square root or the \(1/N\) only rescales everything uniformly and never changes the within-row order. This problem states the equivalent "per-column mean square, then square root" form.

Three details that are easy to get wrong:

1. **k truncates, it does not round**: upstream uses `int(W_metric.shape[1] * sparsity)`, so `d_in=7, sparsity=0.5` prunes 3, not 4;
2. **ties go to the smaller column index**: upstream takes the first k of `torch.sort(..., stable=True)` in ascending order, so equal scores are broken by column index;
3. **structured mode only accepts 0.5**: with `prune_n != 0` every block of `prune_m` consecutive columns keeps only its `prune_n` largest scores; `main.py` asserts `sparsity == 0.5` for `2:4`/`4:8`, and this problem also requires `in_features % prune_m == 0` (upstream `topk` on a ragged tail block is not well defined).

## Task

Implement `solve(weight, activations, sparsity, prune_n=0, prune_m=0)`:

- `weight`: `(out_features, in_features)`, float32/float64;
- `activations`: `(N, in_features)`, same dtype, `N >= 1`;
- `sparsity`: numeric in `[0, 1]` (bool is not a valid scalar);
- `prune_n`, `prune_m`: non-negative integers; when `prune_n > 0`, require `0 < prune_n < prune_m`, `in_features % prune_m == 0`, and `sparsity == 0.5`, otherwise raise `ValueError`.

Compute `metric = |weight| * sqrt(mean_N(activations**2))` in float64. Selection:

- `prune_n == 0`: per row, stable-sort the metric ascending and prune the first `int(in_features * sparsity)`;
- `prune_n > 0`: per row, inside every consecutive block of `prune_m` columns, prune the `prune_n` smallest.

Break ties by smaller column index. Return a 4-key dict:

- `metric`: `(out, in)` in the input dtype;
- `mask`: `(out, in)` bool, **True means pruned**;
- `pruned_weight`: `(out, in)` with zeros at pruned positions and original values elsewhere, in the input dtype;
- `sparsity`: python float, the achieved ratio `mask.sum() / numel`.

Do not use `torch.nn.utils.prune`; do not mutate the inputs.

## Example

```
weight      = [[ 1.0, -5.0, 3.0, -2.0],
               [ 0.5,  9.0,-2.0, -8.0]]
activations = [[ 2.0,  1.0, 0.0,  1.0],
               [ 0.0,  3.0, 0.0,  0.0]]
sparsity    = 0.5            # k = int(4 * 0.5) = 2

column mean square = [2, 5, 0, 0.5]     column RMS = [1.4142, 2.2361, 0, 0.7071]
metric             = [[1.4142, 11.1803, 0, 1.4142],
                      [0.7071, 20.1246, 0, 5.6569]]
```

- Row 0: column 2 is smallest (0), then columns 0 and 3 **tie** at `1.41421356…`; the smaller index 0 is pruned → `mask=[T,F,T,F]`;
- Row 1: column 2 (0), then column 0 (0.7071) → the same `[T,F,T,F]`.

```
pruned_weight = [[0.0, -5.0, 0.0, -2.0],
                 [0.0,  9.0, 0.0, -8.0]]
sparsity      = 0.5
```

Column 2 holds weights 3 and -2, which do not look small, but its input activation is always zero, so its RMS is zero and both rows drop it first — exactly where Wanda departs from pure magnitude pruning.

## Constraints

- **Aggregate activations per column**: RMS averages over tokens and broadcasts across rows; it is neither a row statistic nor an elementwise activation product.
- **Truncating k**: `int(in_features * sparsity)`; `7×0.5 → 3`. Using `round` prunes one extra element and diverges from upstream.
- **Ties go to the smaller index**: upstream relies on a stable ascending sort; this problem pins that as a hard rule and tests it with all-equal weights and activations.
- **N:M constraints**: `sparsity` must be exactly 0.5; selection is per block and blocks never compete with each other.
- **Mask-only edit**: unpruned entries keep their values and signs; `metric` and `pruned_weight` use the input dtype while all arithmetic is float64.
