# Distributed Gradient Clipping (Global Norm)

## Problem

Gradient clipping (clip_grad_norm) caps the overall gradient norm at a threshold to prevent gradient explosion. In a distributed setting the parameters are **sharded** across ranks, so clipping must use the **global norm** uniformly — **ranks must not clip their own shards independently**, since that gives different scaling factors and is equivalent to changing the update direction. This problem is exactly that "easiest thing to get wrong".

## Rules

- Implement `solve(shards, max_norm, norm_type=2.0)`:
  - `shards`: a list of length $W$, where `shards[r]` is the one-dimensional gradient shard on rank `r` (ranks **may have different lengths**);
  - Treat all shards as one big vector and compute the **global $p$-norm**:
    $$ \|\mathbf{g}\|_p = \left(\sum_{r}\sum_{i} |g_{r,i}|^{\,p}\right)^{1/p} $$
  - The scaling factor is $\text{scale} = \min\!\left(1.0,\ \dfrac{\text{max\_norm}}{\|\mathbf{g}\|_p}\right)$
    (when the norm is $0$ or already $\le$ `max_norm`, scale=1 and no clipping occurs);
  - **Every shard is multiplied by the same scale**: $g_{r,i} \leftarrow g_{r,i}\cdot \text{scale}$.
- Return `(clipped, total_norm)`: `clipped` is a `list[torch.Tensor]` of the same length as the input, and `total_norm` is the global norm (float).
- Do not modify the input tensors.

## Examples

```
Input: shards = [ [3.0, 4.0], [0.0] ], max_norm=2.5, norm_type=2.0
  total_norm = sqrt(3² + 4² + 0²) = 5.0
  scale = min(1, 2.5/5.0) = 0.5
Output: ( [ [1.5, 2.0], [0.0] ], 5.0 )
```

> Note: if each rank incorrectly clipped its own shard independently, rank0's local norm 5 would be clipped while rank1's local norm 0 would not — different from the global result.

## Constraints

- `1 ≤ W ≤ 64`; shards are one-dimensional, float, possibly of differing lengths; `max_norm ≥ 0`; `norm_type ≥ 1` (this problem uses finite $p$, excluding $\infty$)
- Return `(list[torch.Tensor], float)`; `total_norm` and the clipped tensors are checked with a relative tolerance of `1e-5`
