# AdaKV: Cross-Head Flattened Global Top-K + Per-Head Safeguard

## Problem

Plain `ScorerPress` lets each KV head pick its own top-k—every head keeps the same number of tokens. **AdaKV** ([arXiv:2407.11550](https://arxiv.org/abs/2407.11550)) observes that the number of important tokens differs greatly across heads, and letting heads "compete for budget" saves more memory. The approach **flattens the scores across heads**, sorts globally, and prunes the $H \times (L - n_\text{kept})$ lowest scores.

To prevent any head from being hollowed out, a **per-head safeguard** is added: each head first forcibly keeps its $\lfloor n_\text{kept} \cdot \alpha \rfloor$ highest scores (pushing those scores to $+\infty$), after which global pruning proceeds.

## Rules

Given `scores` $(H, L)$, `compression_ratio` $\in [0,1)$, `alpha_safeguard` $\in [0,1]$:

- $n_\text{kept} = L - \lfloor L \cdot \text{compression\_ratio} \rfloor$ is the **number kept** (equivalent to `int(L * (1 - cr))`)
- $n_\text{pruned} = H \cdot (L - n_\text{kept})$ — this many are pruned globally
- If `alpha_safeguard > 0`: $n_\text{safe} = \max(1, \lfloor n_\text{kept} \cdot \alpha \rfloor)$; for each head, set the top-$n_\text{safe}$ scores to $+\infty$ (`torch.finfo(float32).max`)
- From the flattened scores, pick the $n_\text{pruned}$ **smallest** and prune them; the rest are kept
- Return a bool mask `kept` of shape $(H, L)$

## Examples

```
scores = [[0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
          [0.6,  0.5, 0.4,  0.3, 0.2,  0.1]]
compression_ratio = 0.5  →  n_kept = 3, n_pruned = 2*(6-3) = 6

alpha_safeguard = 0.5  →  n_safe = max(1, floor(3*0.5)) = 1
  protect (0,0) and (1,0) to +∞
  from the remaining 10 non-∞ scores pick the 6 smallest → (0,5),(0,4),(0,3),(1,5),(1,4),(1,3)
  kept = [[T,T,T,T,T,F],
          [T,F,F,F,F,F]]   ← head 1 keeps only 1 (rescued by the safeguard)

alpha_safeguard = 0.0  →  no protection
  global smallest 6 → head 1 is pruned entirely: [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], head 0 only loses 0.7]
  kept = [[T,T,T,T,T,T],
          [F,F,F,F,F,F]]
```

## Constraints

- CPU float32; $\lfloor\cdot\rfloor$ uses Python `int()` (truncation toward zero).
- `topk` picks the largest; `bottom-k` via `topk(-x)` or `topk(..., largest=False)`.
- The safeguard value uses `torch.finfo(torch.float32).max` rather than `float('inf')` (consistent with kvpress).
