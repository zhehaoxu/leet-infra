# CriticalKV: Two-Stage Wo·V Weighted Retention

## Problem

H2O/SnapKV-style scores only ask "how much attention a key received", yet whether a token is truly *critical* also depends on how much its **value content** contributes after passing through the attention output projection \(W_o\). CriticalKV (Li et al., 2025, [arXiv:2502.03805](https://arxiv.org/abs/2502.03805); `CriticalKVPress` in `kvpress/presses/criticalkv_press.py` of NVIDIA/kvpress) wraps any base scorer into a **two-stage selection**:

- **Stage 1**: protect a small set of tokens chosen by the base score;
- **Stage 2**: the remaining candidates are re-ranked by a new score — the base score (plus a smoothing term) times the L1 norm \(\|v_t W_o\|_1\) of the token's value after the output projection;
- positions picked in stage 1 are overwritten with the dtype's maximum float, so they can never be evicted in stage 2.

## Rules

Inputs: base scores of shape \((H, L)\) per kv head; `wov_norm` of the same shape, i.e. each token's \(\|v_t W_o\|_1\) averaged over the query heads of one GQA group; scalars `compression_ratio` \(r\in[0,1)\), `first_stage_ratio` \(s\in[0,1]\), `epsilon` \(\varepsilon\ge 0\).

1. Total budget \(n\_kept=\max(1,\lfloor L(1-r)\rfloor)\).
2. Stage-1 budget \(b_1=\lfloor L(1-r)\,s\rfloor\); if \(s>0\) then \(b_1=\max(1,b_1)\) (the upstream guard that never lets a short context skip stage 1 entirely), and \(b_1\le L\).
3. Stage 1: per row, take the top-\(b_1\) base scores (larger first, **ties go to the smaller position**).
4. Stage-2 final score: \(\text{final}=(\text{base}+\varepsilon)\cdot \text{wov\_norm}\), then overwrite the stage-1 positions with the float32 ceiling \(3.4028235\times10^{38}\).
5. Per row, take the top-\(n\_kept\) final scores (ties go to the smaller position) and output the kept positions **in ascending order**.

Return two tensors: `kept` (int64, shape \((H, n\_kept)\)) and the final score matrix (float32, shape \((H, L)\)).

## Example

```
base row0 = [0.9, 0.2, 0.7, 0.1, 0.5, 0.3]   wov_norm row0 = [1.0, 3.0, 0.5, 2.0, 4.0, 0.2]
base row1 = [0.4, 0.8, 0.1, 0.6, 0.2, 0.9]   wov_norm row1 = [2.0, 0.1, 5.0, 0.5, 1.0, 0.3]
r = 0.5, s = 0.5, eps = 1e-4  →  n_kept = 3, b1 = 1

row0: stage-1 hit = {0};
  final = [max, 0.6003000, 0.3500500, 0.2002000, 2.0004001, 0.0600200]   (max = 3.4028235e+38)
  top-3 → kept = [0, 1, 4]
row1: stage-1 hit = {5};
  final = [0.8002000, 0.0800100, 0.5005000, 0.3000500, 0.2001000, max]
  top-3 → kept = [0, 2, 5]
```

## Constraints

- CPU float32; `wov_norm` is non-negative; \(0\le r<1\), \(0\le s\le 1\).
- **Invariant**: the stage-1 set is always a subset of the final kept set (\(b_1\le n\_kept\) holds by construction).
- With \(s=0\) the method degenerates to pure value-projection-weighted top-k (no protection at all); with \(r=0\) every row keeps all \(L\) positions.
- All ties are broken by "smaller position first"; the kept indices are sorted ascending before output.
