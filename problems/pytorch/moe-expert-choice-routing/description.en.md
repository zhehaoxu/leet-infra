# MoE Routing: Expert-Choice Routing

## Problem

Shazeer 2022, "Mixture-of-Experts with Expert Choice Routing", reverses the routing direction: instead of each token picking $k$ experts, **each expert picks the $k_c$ tokens it wants most**. This gives perfect load balancing for free (each expert is assigned exactly $k_c$ tokens).

## Rules

Implement `solve(logits, k_c)`, where `logits` has shape $(T, E)$:

1. `probs = softmax(logits, -1)` (normalized over the expert dimension for each token).
2. For each expert $e$: take the $k_c$ tokens with the largest `probs[:, e]` along the $T$ dimension (ties broken in favor of the smaller token index).
3. `mask (T, E) bool`: `mask[t, e] = True` if and only if token $t$ is selected by expert $e$.
4. `weights = probs * mask.float()`; renorm over the $E$ dimension to get `weights_renormed`.
5. **If no expert selects a token** (`mask[t].sum()==0`), its renormalized row is all zeros (denominator protection: `clamp_min(1e-30)` or `torch.where`).

Return `(mask, weights_renormed)`.

## Examples

```
logits = [[2, 1, 0],
          [0, 2, 1],
          [1, 0, 2],
          [3, 3, 0],
          [0, 0, 1],
          [-1, -1, -1]]
k_c = 2

mask =    [[1,0,0],
           [0,1,0],
           [0,0,1],
           [1,1,0],
           [0,0,1],
           [0,0,0]]   # token 5 is selected by 0 experts
weights = [[1,0,0],
           [0,1,0],
           [0,0,1],
           [0.5,0.5,0],
           [0,0,1],
           [0,0,0]]
```

## Constraints

- CPU float32; `mask.dtype == torch.bool`; `mask[:, e].sum() == k_c` holds for every column.
