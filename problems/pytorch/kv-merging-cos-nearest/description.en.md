# Merging: Fold Evicted Tokens' Values into Their Most Similar Survivor

## Problem Description

Plain top-k pruning **hard-drops** the values of low-score tokens — the information is lost forever. **MergingPress** proposes a soft landing: **weighted-merge** each evicted token's value into "the most similar survivor" (similarity measured by cosine of the keys), preserving the information while keeping the shape unchanged. The idea comes from Token Merging (ICLR 2023) and D2O.

The merge weight $w$ has two components:
- **Similarity gate**: $\max(\text{sim}, 0)$ (the merge is discarded if the nearest neighbor's cosine is negative)
- **Norm ratio**: $\|v_\text{evict}\| / (\|v_\text{evict}\| + \|v_\text{target}\| + \varepsilon)$ — a large value merged into a small one gets a high weight, and vice versa it gets diluted

The new value of the target kept slot uses a **weighted average**: $v_\text{new} = (v_\text{kept} + \sum w \cdot v_\text{evict}) / (1 + \sum w)$. If some kept slot is not chosen by any evict, its original value stays unchanged.

## Rules

Given `keys` $(H, L, D)$, `values` $(H, L, D)$, `scores` $(H, L)$, `n_kept`:

1. `kept_idx = scores.topk(n_kept, largest=True).indices` — shape $(H, n\_kept)$, **sorted by descending score**
2. `evict_idx` is the complement of kept, in original sequence order $(H, L - n\_kept)$
3. Normalize both key groups (divide by L2 norm, clamp min = $\varepsilon$)
4. `sim = ev_keys_normalized @ kept_keys_normalized^T` → $(H, n\_evict, n\_kept)$
5. For each evict, pick `max_sim, target_slot` (max along $n\_kept$)
6. $w = \max(\text{sim}, 0) \cdot \dfrac{\|v_\text{evict}\|}{\|v_\text{evict}\| + \|v_\text{target}\| + \varepsilon}$
7. `vacc[slot] += w * v_evict`; `wacc[slot] += w` (scatter_add)
8. `updated[slot] = (v_kept[slot] + vacc[slot]) / (1 + wacc[slot])` if `wacc[slot] > 0`; otherwise `v_kept[slot]`
9. **Scatter back into the original sequence**: only rewrite the values at kept positions; the values at evicted positions keep their original values (unchanged)

Return a float32 `merged_values` of shape $(H, L, D)$. Fix $\varepsilon = 10^{-6}$, `similarity_threshold = 0`, `merge_fraction = 1`.

## Example

```
H=2, L=4, D=2, n_kept=2
scores = [[0.9, 0.1, 0.7, 0.2],   [0.3, 0.4, 0.8, 0.1]]

h=0: kept_idx=[0, 2] (scores 0.9, 0.7), evict=[1, 3]
  Both evicts are most similar to kept slot 0 → slot 0's value is rewritten, slot 1 is kept
  merged values[0] = [1.814293, 2.814292]
  merged values[2] = [5.717330, 6.717330]

h=1: kept_idx=[2, 1] (scores 0.8, 0.4), evict=[0, 3]
  Both evicts are most similar to kept slot 1 (original position 1) → slot 0 stays unchanged
  merged values[2] = [2, 3] (no contribution)
  merged values[1] = [1.821373, 2.124212]

result = [[[1.814293, 2.814292], [3.0, 4.0], [5.717330, 6.717330], [7.0, 8.0]],
          [[0.5, 0.5], [1.821373, 2.124212], [2.0, 3.0], [4.0, 5.0]]]
```

## Constraints

- CPU float32; $\varepsilon = 10^{-6}$.
- `torch.topk` outputs in descending score order, ties broken by ascending index.
- The order of `evict_idx` is ascending original position (`evict_mask.nonzero()[:, 1].reshape(H, n_evict)`).
