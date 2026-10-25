# Softmax (Numerically Stable)

## Problem

Implement a numerically stable Softmax along a given dimension, normalizing a set of logits into a probability distribution. It is the core component of Attention:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Computing `exp` directly on inputs at the `logits=1000` scale overflows to `inf` (float32 already overflows at $e^{88}$). The key trick: **multiply numerator and denominator by $e^{-\max z}$** — subtract the maximum along that dimension before normalizing. The result is mathematically unchanged, but the argument of `exp` is always $\le 0$ so it can never overflow.

## Rules

- Implement `solve(logits: torch.Tensor, dim: int = -1) -> torch.Tensor`.
- Apply the stable softmax (subtract-max trick) along `dim`; every slice along that dimension must sum to 1.
- Output shape matches the input; vectorized implementation; no `NaN`/`inf` may appear (including inputs at the `logits=1000` scale).

## Example

```
Input: [[1.0, 2.0, 3.0],
       [0.5, 0.5, 0.5]]   (dim=-1)
Output: [[0.09003, 0.24473, 0.66524],
       [0.33333, 0.33333, 0.33333]]
```

## Constraints

- Input is a float32 or float64 tensor of any shape (at least 2 dimensions)
- `dim` is a valid dimension (it may be negative)
- Values may reach ±1000
