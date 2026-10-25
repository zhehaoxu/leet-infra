# Approach

## Interviewer's Intent

"How to do gradient clipping correctly under data parallelism / ZeRO" is the watershed between "has used clip_grad_norm_" and "understands its internal logic". Most people have no trouble with the single-machine version, but as soon as parameters are sharded they err by **clipping with each GPU's local norm** — this problem forces you to realize that the clipping scale must come from the **global norm**.

## Why Ranks Cannot Clip Independently

The essence of clipping is: if the overall gradient is too long, shrink it **proportionally** to `max_norm` without changing its direction. In a distributed setting the parameters are sharded across ranks, and the "overall gradient" is the big vector formed by concatenating all shards. Therefore:

1. First compute the global norm $\|\mathbf{g}\|_p = \big(\sum_r\sum_i |g|^p\big)^{1/p}$ — this is a **cross-rank reduce** (each GPU computes $\sum|g|^p$ over its own shard, All-Reduce sums them, then take the $p$-th root).
2. The globally unique $\text{scale}=\min(1,\ \text{max\_norm}/\|\mathbf g\|_p)$.
3. Every GPU clips its own shard with **the same scale**.

If each GPU computes its scale from its own shard's local norm, the scaling factors differ → the norm of the reassembled whole is neither `max_norm` nor the original value, and **the direction is corrupted**. `check_global_not_per_rank_independent` pins this down with a two-GPU example (local 0.6/0.8, global exactly 1.0): the correct answer is for both GPUs to multiply by 0.5, not 0.5/0.6 and 0.5/0.8 respectively.

## Correspondence with the torch Implementation

Inside `torch.nn.utils.clip_grad_norm_`: the `.grad` of all parameters are concatenated into total_norm, then `clip_coef = max_norm / (total_norm + 1e-6)`, `clip_coef_clamped = clamp(clip_coef, max=1.0)`, and each is multiplied back. This problem omits the `+1e-6` in favor of `min(1, max_norm/norm)` plus an explicit zero guard, which is semantically equivalent and keeps the example numbers clean. With sharding, replace the single-machine norm in `total_norm` with the cross-GPU sum from step 4 (exactly the standard way to do global clipping in `torch.distributed`).

## norm_type and Edge Cases

- $p=2$ is the Euclidean norm (default); $p=1$ is the sum of absolute values; $\infty$ is also common in practice (not included here).
- `total_norm==0` → scale=1 (otherwise 0/0 produces NaN).
- Already `<= max_norm` → scale=1, unchanged.

## Grading Approach

`test_cases.py` checks against an independent reference that "first `cat`s everything into a big vector and then computes the norm" (a different derivation path from summing over shards), covering: a discriminative case of global vs independent clipping, no clipping below the threshold, $p=1$, zero gradients, **shards of unequal length**, and the return structure `(list, float)` plus no modification of inputs.
