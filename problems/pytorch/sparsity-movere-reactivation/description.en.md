# Sparsification: Dynamic Sparse Training Drop-Grow Reactivation

## Problem

Fixed-mask pruning has a structural flaw: whatever is pruned at initialisation can **never** come back, even if later gradients show it was important. Dynamic sparse training (DST) therefore lets the mask change every step: **drop the least important active elements and give the slots to the inactive elements with the largest gradients**. RigL's rule (Evci et al., ICML 2020) is exactly "drop by weight magnitude, grow by gradient magnitude"; MovERec-style work adds momentum and a reactivation rate \(\varepsilon\) for smoother replacement.

This problem performs a **single** update and makes randomness explicit: no random numbers are sampled, a supplied array of uniform values `u` is used only to break ties.

## Rules

Given `mask`, `weight`, `grad`, `u` (float32, same shape, treated via flattened indices; `u∈[0,1)`), `sparsity` (target sparsity) and `eps` (reactivation rate \(\varepsilon\)):

Target active count \(k=N-\text{round}(\text{sparsity}\cdot N)\) with \(N\) the number of elements.

1. **Reach the target**:
   - if active count \(<k\): reactivate \(k-\)active elements from those **inactive in the input mask**, ranked by \(|grad|\) **descending** (ties by \(u\) descending, then index ascending);
   - if active count \(>k\): prune the surplus by \(|weight|\) **ascending** (ties by index ascending).
2. **Swap**: \(r=\text{round}(\varepsilon\cdot\text{current inactive count})\), capped by the current active count and the candidate-pool size;
   - first prune the \(r\) active elements with the smallest \(|weight|\);
   - then reactivate \(r\) elements from the candidate pool (inactive in the input mask and not grown in step 1), ranked by \(|grad|\) descending (ties by \(u\) descending, then index ascending).

Return `(new_mask, reactivated, pruned)`: `new_mask` is float32 0/1 with the input shape; the other two are int64 flattened indices in **ascending** order.

## Examples

```
mask   = [1, 1, 1, 1, 0, 0]
weight = [3, -1, 0.5, 2, -0.2, 4]
grad   = [0.1, 2, -3, 0.4, 5, -1]
u      = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
sparsity = 1/3, eps = 0.5

k = 6 - round(2) = 4        # active count is already 4, nothing to fix
pool = {4, 5}, current inactive count = 2
r = round(0.5 * 2) = 1

drop: active |weight| = [3, 1, 0.5, 2] -> the smallest is index 2 -> pruned = [2]
grow: pool |grad| = [5 (idx4), 1 (idx5)] -> pick 5 -> reactivated = [4]

new_mask = [1, 1, 0, 1, 1, 0]
```

## Constraints

- CPU float32; `mask` counts as active where `>0`; `u` only breaks ties and must not drive extra random sampling.
- After the update the active count must be exactly \(k\) (when the input provides enough candidates).
- **Reactivated ⊆ input-inactive set; pruned ⊆ input-active set; the two are disjoint** — an element dropped in step 1 because of overshoot is never grown back in the same round.
- `eps=0` performs only the "prune to k" step and reactivates nothing.
