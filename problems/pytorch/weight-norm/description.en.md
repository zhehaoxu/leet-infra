# WeightNorm Reparameterization

## Problem

Weight Normalization splits the weight $w$ into two independent parameters: a **direction** $v$ and a
**length** $g$:

$$
w = g \cdot \frac{v}{\|v\|_2},\qquad \|v\|_2 = \sqrt{\textstyle\sum_{\text{slice}} v_i^2}
$$

Cutting the tensor along one axis (`dim`) yields several "slices" (for example, each output channel of a
convolution kernel), each decomposed independently: the norm of a slice is exactly $|g|$, while the direction
keeps the direction of $v$. The optimizer is better conditioned on the $\log$-scale $g$.

## Rules

- Implement `solve(weight_g: torch.Tensor, weight_v: torch.Tensor, dim: int = 0) -> torch.Tensor`, returning
  the expanded equivalent weight $w$, with the same shape as `weight_v`.
- The norm is computed along `dim` with **`keepdim=True`** (the shape of `g` is that of `v` with `dim`
  collapsed to 1), which guarantees per-slice broadcasting.
- **Division-by-zero pitfall**: when a slice is all zeros, $\|v\|=0$. PyTorch's implementation applies
  `clamp_min(1e-12)` to the denominator, so an all-zero slice outputs 0 and must never produce NaN.
- The input tensors must not be modified in place.

## Example

```
Input: g = [[1.0], [2.0]], v = [[3,4], [3,4]], dim = 1
Norm of each row: ||(3,4)|| = 5
Output: [[0.6, 0.8], [1.2, 1.6]]     # g * v / 5
```

## Constraints

- float32/float64; `dim` is a valid dimension
- Matches the weight produced by `torch.nn.utils.weight_norm` after expansion; non-zero slices satisfy
  $\|w_{\text{slice}}\| = |g_{\text{slice}}|$
