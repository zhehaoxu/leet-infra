# PEFT: WaveFT Sparse Spectrum Delta Weights

## Problem

**WaveFT** (peft `tuners/waveft.py`) does not learn the weight matrix; it learns the weight's **spectrum coefficients**. The delta \(\Delta W\) is viewed as a set of transform coefficients and training only updates \(n_\text{frequency}\) randomly chosen positions, with everything else pinned to zero. Unlike LoRA's *low-rank* constraint this is a **coordinate-sparsity** constraint: the parameter count is \(n_\text{frequency}\), independent of any rank, and can be far below \(r(d_\text{out}+d_\text{in})\).

This problem covers only the **direct path** of peft's `get_delta_weight` (`use_idwt=False`): the spectrum values are scattered straight into a dense matrix shaped like \(W_0\), scaled by `scaling`, and the adapter forward simply adds \(x\Delta W^\top\) on top of the base output.

## Rules

- Positions come from the seed: `torch.Generator().manual_seed(random_loc_seed)` with `torch.randperm(d_out*d_in)[:n_frequency]` for the flat indices, mapped by `row = idx // d_in`, `col = idx % d_in` (row-major flattening).
- \(\Delta W[\text{row},\text{col}] = \text{spectrum}[k]\), and the whole matrix is then multiplied by `scaling`.
- Forward: \(y = xW_0^\top + b_0 + x\Delta W^\top\); when `b0` is None the bias term is dropped.
- Compute in float64, return float32.

## Example

\(W_0(3\times4)\), \(n_\text{frequency}=4\), seed 777, `scaling=25.0`:

```
Input:
  W0 = [[0.5,-0.2,0.1,0.3],[0.2,0.4,-0.1,0.0],[-0.3,0.1,0.5,0.2]]
  b0 = [0.1,-0.2,0.05]
  x = [[1.0,0.5,-0.3,0.2],[0.1,-0.4,0.6,0.3]]
  spectrum = [0.02,-0.01,0.03,0.015]
Output:
  delta = [[0.0,0.0,0.0,0.75],[0.0,0.0,0.0,-0.25],[0.375,0.0,0.0,0.5]]
  y     = [[0.68,0.18,0.165],[0.605,-0.475,0.5275]]
```

## Constraints

- `1 <= n_frequency <= d_out*d_in` and `spectrum` is 1-D with length exactly `n_frequency`; out-of-range or malformed shapes raise `ValueError`.
- `x` may have any leading dimensions; its last dimension is \(d_\text{in}\).
- Degenerate properties: an all-zero spectrum yields a strictly zero delta and the base forward; with `n_frequency = d_out*d_in` every cell receives exactly one coefficient (the delta is a permutation of the spectrum).
- The `use_idwt=True` inverse-wavelet path is out of scope here (peft defaults to IDWT, which needs `waverec2d`).
