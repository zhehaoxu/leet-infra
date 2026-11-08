# Captum Occlusion's sliding-window perturbation attribution ledger

## Background

`captum.attr.Occlusion` is a **black-box perturbation-based attribution** method. Unlike Integrated Gradients (which integrates along a path) or Layer-wise Relevance Propagation (which propagates rules layer by layer), it slices the input into sliding windows, replaces the features inside each window with a baseline, and measures how the model output changes.

Its ledger has four parts:

1. **Grid enumeration**: how many times the window shifts along each dimension, how many windows in total, and each window's offset.
2. **Window masks and coverage counts**: which feature positions each window actually covers, and how many windows cover each position.
3. **Difference ledger**: the original output, the output after each window is ablated, and the difference between them.
4. **Attribution ledger**: accumulate each window's difference onto the positions it covers, then divide by the coverage count to get the final attribution.

Your task is to rebuild this whole ledger in pure Python, while getting right several details that you would get wrong unless you read the official source.

## Conventions

- All inputs are **exact rational numbers** (computed internally with `fractions.Fraction`); conversion to `float` happens only at the final output.
- The forward model is an explicit MLP: `layers` is a list of layers, each of the form `{"W": [[...]], "b": [...]}`. `W`'s row count is the output dimension and its column count is the input dimension. **ReLU is applied between layers, but not after the last layer.**
- The input `x` may be a nested list (shaped like `shape`); it is flattened internally before any computation.
- `shape` describes the **non-batch** dimensions; every dimension must be a positive integer.
- `window_shape` (the sliding window shape) must have the **same number of dimensions** as `shape`, each a positive integer, and must **not exceed** the corresponding input dimension.
- `strides` may be:
  - `None` → every dimension uses stride `1`;
  - a **scalar integer** → that value applies to every dimension;
  - a list with the **same number of dimensions** as `shape`, one positive integer per dimension.
- The stride constraint (same as the official implementation): **a dimension's stride must not exceed that dimension's window length**, unless the window covers the whole input dimension exactly (i.e. `shape[d] == window_shape[d]`).
- `baseline` may be a scalar (broadcast to every position), or a nested/flat list with the same shape as `shape` / the same length after flattening.
- `target` is an index into the output dimension, using **Python/PyTorch negative-index semantics**: `target=-1` selects the last output and `target=-n` selects the first; out-of-range values (`target >= n` or `target < -n`) raise an error.

## mode 1: `occlusion_grid` — the grid enumeration ledger

```python
solve("occlusion_grid", shape=[5], window_shape=[3], strides=[1])
```

Outputs:

```json
{
  "shift_counts": [3],
  "num_windows": 3,
  "offsets":  [[0], [1], [2]],
  "extents":   [[[0, 3]], [[1, 4]], [[2, 5]]]
}
```

- `shift_counts[d]`: the number of shifts along dimension `d`, equal to `ceil((shape[d] - window_shape[d]) / strides[d]) + 1`.
- `num_windows`: the product of the per-dimension shift counts.
- `offsets[k]`: the starting offset of window `k` in each dimension (**one list per dimension**).
- `extents[k]`: the **actual covered interval** `[start, stop)` of window `k` in each dimension (**nested twice**: first by dimension, then by interval endpoints).

### ★ Enumeration order: base-conversion, dim0 is the least significant digit

The official `_occlusion_mask` enumerates windows like this:

```python
for idx in range(num_windows):
    remaining = idx
    current_index = []
    for i, shift_count in enumerate(shift_counts):
        current_index.append((remaining % shift_count) * strides[i])
        remaining = remaining // shift_count
```

That is, **dimension 0 changes fastest** (it is the least significant digit). For `shape=[3,4]`, `window_shape=[2,2]`, `strides=[1,2]`, we get `shift_counts = [2, 2]` and the enumeration order is:

```
(0,0) -> (2,0) -> (0,2) -> (2,2)
```

not the row-major order `(0,0) -> (0,2) -> (2,0) -> (2,2)`.

### ★ Window truncation: the last window gets clipped

When a dimension's window starts at `start` and has length `w`, and `start + w > shape[d]`, the official implementation truncates via **negative padding** in `torch.nn.functional.pad` — a negative right padding clips the tail of the window off. This is equivalent to clamping the right end of the covered interval to `shape[d]`.

With `shape=[6]`, `window_shape=[4]`, `strides=[3]`:

- `shift_counts = ceil((6-4)/3)+1 = 1+1 = 2`
- Window 0: offset 0 → covers `[0, 4)`
- Window 1: offset 3 → `3+4 = 7 > 6` → truncated to `[3, 6)`

## mode 2: `occlusion_mask` — the mask and coverage ledger

```python
solve("occlusion_mask", shape=[3, 4], window_shape=[2, 2], strides=[1, 2])
```

Outputs:

```json
{
  "masks":   [[0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0], ...],
  "coverage": [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1]
}
```

- `masks[k]`: the 0/1 mask of window `k`, **flattened in row-major order**, of length `prod(shape)`.
- `coverage[i]`: how many windows cover feature position `i` (**the number of windows**, not the sum of coverages).

The flattening order of the multi-dimensional masks is row-major: the last dimension varies fastest.

## mode 3: `occlusion_diff` — the difference ledger

```python
solve("occlusion_diff",
      layers=[{"W": [[1.0, 2.0, 3.0, 4.0, 5.0]], "b": [0.5]}],
      x=[1.0, 2.0, 3.0, 4.0, 5.0],
      baseline=0.0,
      shape=[5], window_shape=[3], strides=[1], target=0)
```

Outputs:

```json
{
  "initial_eval": 55.5,
  "window_evals": [41.5, 26.5, 5.5],
  "eval_diffs":   [14.0, 29.0, 50.0],
  "total_attrib": [14.0, 43.0, 93.0, 79.0, 50.0]
}
```

- `initial_eval`: the forward output on the original input (the `target` slice).
- `window_evals[k]`: the forward output after replacing the features inside window `k` with the baseline.
- `eval_diffs[k]`: `initial_eval - window_evals[k]`.
- `total_attrib[i]`: the **unnormalized** accumulation — the sum of the `eval_diffs` of every window that covers position `i`.

Ablation is applied per position: `ablated[i] = x[i] * (1 - mask[i]) + baseline[i] * mask[i]`, i.e. **only covered positions** are replaced by the baseline.

## mode 4: `occlusion_attribution` — the full attribution ledger

```python
solve("occlusion_attribution",
      layers=[{"W": [[1.0, 2.0, 3.0, 4.0, 5.0]], "b": [0.5]}],
      x=[1.0, 2.0, 3.0, 4.0, 5.0],
      baseline=0.0,
      shape=[5], window_shape=[3], strides=[1], target=0)
```

Outputs:

```json
{
  "total_attrib": [14.0, 43.0, 93.0, 79.0, 50.0],
  "weights":     [1, 2, 3, 2, 1],
  "attribution": [14.0, 21.5, 31.0, 39.5, 50.0]
}
```

### ★★ Overlapping regions are averaged (the single most important discriminator here)

`attribution[i] = total_attrib[i] / weights[i]`, where `weights[i]` is the **number of windows** covering position `i`.

This differs from `FeatureAblation` (`use_weights=False`), which accumulates every window's difference without dividing by the coverage count.

In the example above, position 2 is covered by 3 windows (`weights[2] = 3`), the accumulated value is `93.0`, and the attribution is `93.0 / 3 = 31.0`.

### ★ float32 rounding

The official `format_result` ends with:

```python
single_attrib = single_attrib.float() / weight
```

That is, **the float64 accumulated value is first demoted to float32, then the division is performed in float32**. Your output must reproduce this rounding: when `total_attrib[i] / weights[i]` is not a binary finite decimal (e.g. `1/3`), the expected value is the float32-rounded `0.3333333432674407958984375`, not the float64 `0.3333333333333333`.

## Common mistakes

1. **Enumerating in row-major order.** The official order is base-conversion, with dim0 as the least significant digit.
2. **Forgetting the last-window truncation.** Negative padding in `F.pad` clips the window tail, so the right end of `extents` must be clamped to `shape[d]`.
3. **Not averaging the overlapping regions** (`attribution` equals `total_attrib`), or dividing by the total window count instead of the number of windows covering that position.
4. **Emitting the plain float64 quotient** instead of reproducing the official float32 demotion.
5. **Flipping the sign of `diff`.** `eval_diffs[k] = initial_eval - window_evals[k]`, so the difference is positive when ablation *lowers* the output.
6. **Applying the baseline outside the window.** Only covered positions are replaced.
7. **Applying ReLU after the last layer.** The official-equivalent forward applies ReLU between layers and not after the last one.
8. **Ignoring the stride constraint**: when a dimension's window does not cover it exactly, the stride must not exceed the window length.
9. **Not broadcasting a scalar `strides` to all dimensions** (invisible in 1-D, exposed in 2-D).
10. **Rejecting a negative `target` outright.** The official path uses PyTorch's negative-index semantics, so `-1` means the last output.
11. **Flattening multi-dimensional masks in column-major order.** Masks are flattened row-major.
12. **Writing `shift_counts` as `ceil(...)` and forgetting the `+1`**.

## Self-check

```python
solve("occlusion_grid", shape=[6], window_shape=[4], strides=[3])
```

should give `shift_counts = [2]`, `num_windows = 2`, `offsets = [[0], [3]]`, `extents = [[[0, 4]], [[3, 6]]]`.

```python
solve("occlusion_mask", shape=[5], window_shape=[3], strides=[1])
```

should give 3 masks and a coverage count of `[1, 2, 3, 2, 1]`.
