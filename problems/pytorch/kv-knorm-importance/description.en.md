# KNorm: Importance Scores from Key L2 Norms

## Problem Description

**KNorm** (Zhang et al., 2024, [arXiv:2406.11430](https://arxiv.org/abs/2406.11430)): based on the observation that "large attention score ↔ large L2 norm of the key", it directly uses $-\|k\|_2$ as the importance score — the larger the score, the more likely the token is kept, i.e. **the larger the norm, the more important**. This is the cheapest KV scoring method, requiring no attention at all.

In kvpress, `KnormPress.score` is a single line: `return -torch.linalg.vector_norm(keys, dim=-1, dtype=promote(dtype, float32))` — to avoid fp16/bf16 ties, the norm is accumulated in float32 first.

## Rules

Implement `solve(keys)`, where `keys` has shape $(H, L, D)$ (no batch, float32): return a score vector of shape $(H, L)$ with values $-\|k_{h,\ell}\|_2$.

## Example

```
keys = [
  h=0: [[1,2,2], [0,0,0], [3,4,0]],   # norms [3, 0, 5]
  h=1: [[0.5,0.5,0.5], [1,1,1], [2,0,0]]  # norms [√0.75, √3, 2]
]
scores = [
  [-3.0,        -0.0,       -5.0],
  [-0.86602538, -1.7320508, -2.0]
]
```

## Constraints

- The input is already float32 and the return is float32; no explicit dtype promotion is needed.
- $\|0\|_2 = 0$, and negating gives `-0.0`, which is numerically equal to `0.0`.
