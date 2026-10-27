# KV Cache: A2S Adaptive Top-n from the Previous Step's Sparsity Pattern

## Problem

Sparse attention (computing only the top-n keys at each step) is almost the only way to save both memory and compute in long contexts, but two of its implicit assumptions break easily:

1. **n is fixed.** A fixed n wastes work on steps where attention is highly concentrated (3 keys would do, yet 20 are computed) and falls short on steps where it is spread out (20 keys cover only 60% of the mass).
2. **Every step rescores everything.** Selecting the current top-n requires the scores of all keys first, which defeats the purpose.

The A2S family solves both by **reusing the previous step's sparsity pattern as a prior**: the tokens selected last step are the most likely candidates now (attention patterns are strongly correlated between adjacent steps), so the current step only needs to compare those \(k\) candidates and then decide, from the **current accumulated mass**, how many to actually take.

Concretely, given the attention rows \(a_s\in\mathbb R^{L}\) (row \(s\) is valid on its first \(P+s\) columns, \(P\) being the prompt length):

- **Candidate set**: step 0 uses all valid columns; step \(s>0\) uses the top-\(k\) of step \(s-1\) over its valid columns (ties to the smaller index);
- **Candidate ordering**: by the **current** step's scores, descending (ties to the smaller index)—the candidate set comes from the previous step, but the ordering must use current evidence;
- **Adaptive n**: with the current reachable mass \(T=\sum_{j<P+s}a_s[j]\), take the smallest \(m\) such that the current scores of the top \(m\) candidates sum to \(\ge p\cdot T\); if even the full candidate set falls short, \(n=|\text{candidates}|\).

\[
n_s=\min\Big\{m\ \Big|\ \sum_{i=1}^{m} a_s[\pi_i]\ \ge\ p\,T\Big\}
\]

where \(\pi\) is the candidate sequence ordered by current score and \(p\in[0,1]\) is the target mass fraction (typically 0.9).

The **approximation** is the heart of the problem: tokens outside the candidate set are never selected no matter how high their current score is. As soon as the attention pattern jumps between adjacent steps (a new entity, a new instruction, a format change), A2S necessarily misses it—that is the price of saving a full scoring pass.

## Task

Implement `solve(attn, k, p)`:

- Input: `attn` (float32 `(S, L)`, row `s` valid on its first `P+s` columns and zero afterwards, `P = L - S`); `k` (candidate-set size, integer \(\ge 1\)); `p` (target mass fraction in \([0,1]\)).
- Output `(n_adaptive, keep)`: `n_adaptive` is `(S,)` int64 (how many were taken per step); `keep` is `(S, L)` bool.
- The candidate count never exceeds `min(k, valid columns)`; with `p = 0`, \(n=1\) (at least one is taken); a zero-mass step also degenerates to 1.
- Accumulation uses "\(\ge pT\)" with a \(10^{-12}\) floating-point tolerance.
- Raise `ValueError` when `k < 1` or `p ∉ [0,1]`.

## Example

```
L = 6, S = 3 (P = 3), k = 3, p = 0.75
a0 = [0.5, 0.3, 0.2, 0, 0, 0]
a1 = [0.5, 0.3, 0.1, 0.1, 0, 0]
a2 = [0.2, 0.2, 0.2, 0.2, 0.2, 0]

step 0: candidates {0,1,2} (by current score), cumulative 0.5, 0.8 >= 0.75 -> n=2, keep {0,1}
step 1: candidates = previous top3 {0,1,2}, ordered 0,1,2 by current score, cumulative 0.5, 0.8 -> n=2, keep {0,1}
step 2: candidates = previous top3 {0,1,2}, current scores tie at 0.2, cumulative 0.2/0.4/0.6 all short of 0.75 -> n=3, keep {0,1,2}

n_adaptive = [2, 2, 3]
```

## Constraints

- **Candidates come from the previous step, ordering from the current step**: the two must not be mixed. Ordering by the previous step's scores degenerates to "the previous step's top-n" and loses adaptivity; ordering over all current columns means no scoring was saved.
- The mass threshold uses only the **currently valid columns**; including the zeros of invalid columns changes nothing, but treating not-yet-generated columns as candidates is wrong.
- The upper bound of `n` is the candidate count (`min(k, valid columns)`), not `k` and not `L`.
- High-scoring tokens outside the candidate set must stay excluded (approximation); do not widen the candidate set to chase recall.
