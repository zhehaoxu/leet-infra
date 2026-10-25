# MoE Routing: Top-k Gating

## Problem

The **routing gate** of a Mixture-of-Experts (MoE): for each token, pick the $k$ experts with the highest probability out of $E$, **renormalize their gate weights to sum to 1**, and use them as the coefficients for the subsequent weighted combination of expert outputs. This is the `router` forward pass of models such as Mixtral / DeepSeek-V2 / Qwen-MoE.

## Rules

Implement `solve(logits, k)`:
- `logits` has shape $(T, E)$: the raw score of each token for each expert (the linear layer output, before softmax).
- `k` is an integer: how many experts each token selects. $1 \le k \le E$.

Steps:
1. `probs = softmax(logits, dim=-1)` (along the expert dimension).
2. Take the $k$ largest entries of `probs` in each row to obtain `top_vals (T,k)` and `top_idx (T,k)`.
3. **Renormalize**: `scores = top_vals / top_vals.sum(-1, keepdim=True)`.

Return `(scores, indices)`:
- `scores` has shape $(T, k)$, each row summing to 1 (float).
- `indices` has shape $(T, k)$, of type `long`, the expert indices.

> Fully consistent with HF `MixtralTopKRouter.forward`: softmax first, then topk, then renorm. Ties are broken in favor of the smaller expert index.

## Examples

```
logits = [[2.0, 1.0, 0.5, -1.0, -2.0, 0.0],
          [0.0, 0.0, 3.0, 2.4, -1.0, -1.0],
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], k=2

scores = [[0.731059, 0.268941],
          [0.645656, 0.354344],
          [0.731059, 0.268941]]
indices = [[0, 1], [2, 3], [5, 4]]
```

## Constraints

- Everything on CPU, `float32`.
- Grading cross-check: two paths, vectorized `topk` and per-row Python sorting.
