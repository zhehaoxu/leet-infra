# KV Cache: H2O Cumulative Attention and Heavy-Hitter Pool

## Problem

**H2O** (Heavy-Hitter Oracle, Zhang et al., NeurIPS 2023, arXiv:2306.14048) starts from an empirical observation: during decoding, a small number of tokens absorb most of the attention mass, and this set of "heavy hitters" is highly stable across steps. H2O therefore compresses the KV cache into "**sink + cumulative top-k**":

- **sink**: the first few tokens of the sequence (BOS, instruction prefix) are attended to by almost every step and are structurally indispensable, so they are kept unconditionally;
- **heavy hitters**: attention scores of **every step are summed column-wise**, and the tokens with the largest accumulated score are the heavy hitters.

The crucial part is the semantics of "cumulative". Given the attention row of one step \(a_s\in\mathbb R^{L}\) (already summed over heads and queries), H2O maintains

\[
\text{acc}_s[j]=\sum_{t=0}^{s} a_t[j]
\]

with **no normalization at all**—neither dividing by the number of steps seen so far, nor rescaling the row beyond its softmax. The retained set at step \(s\) is

\[
\mathcal K_s=\{\,j<\text{sink}\,\}\ \cup\ \operatorname{top\text{-}k}\big(\text{acc}_s[0:P+s]\big)
\]

where \(P\) is the prompt length and \(P+s\) is the number of columns visible at step \(s\) (the token generated at that step has not been attended to yet, so its accumulated score is always \(0\)). The longer a token survives and keeps being attended to, the larger its accumulated score grows, which is why heavy hitters become increasingly stable.

This problem only asks for the **per-step selection trace**; physical cache compaction and index remapping are out of scope.

## Task

Implement `solve(attn, budget, n_sink)`:

- Input: `attn` (float32 `(S, L)`, `S` steps and `L = P + S` final cache length; row `s` is valid only on its first `P+s` columns, the rest are 0); `budget` (non-negative integer, the number of heavy-hitter slots); `n_sink` (number of sink tokens).
- Output `(acc, keep)`: `acc` is `(L,)` float32 (full-run accumulated column sums); `keep` is `(S, L)` bool (row `s` is the retained mask of step `s`).
- Candidates at step `s` are columns `[0, P+s)`; columns `j < n_sink` are kept unconditionally, the rest take the `budget` largest values of `acc_s`.
- **Ties go to the smaller index** (when `acc_s` values are equal, keep the smaller one).

**Conventions**: accumulate internally in float64 and output float32; `budget` and `n_sink` are integers. Raise `ValueError` when `budget < 0` or `n_sink ∉ [0, L]`.

## Example

```
L = 6, S = 2 (P = 4), budget = 2, n_sink = 1
attn[0, :4] = [0.10, 0.50, 0.20, 0.20]
attn[1, :5] = [0.05, 0.10, 0.25, 0.30, 0.30]

acc = [0.15, 0.60, 0.45, 0.50, 0.30, 0.00]      # the last column is never attended, stays 0
step 0: valid columns 0..3, non-sink scores [0.50, 0.20, 0.20] → take 1, 2 (tie → smaller)
        keep[0] = [T, T, T, F, F, F]
step 1: valid columns 0..4, non-sink scores [0.60, 0.45, 0.50, 0.30] → take 1, 3
        keep[1] = [T, T, F, T, F, F]
```

## Constraints

- `acc` must match "sum the rows, never divide by the step count": with `S` all-one steps the first `P` columns accumulate to `S` (averaging would give 1).
- The token generated at the current step has accumulated score 0, but it becomes a candidate in later steps once attention mass lands on it.
- The number of kept positions per step is exactly `n_sink + min(budget, P+s-n_sink)` (fewer when candidates run out).
- Do not rely on the unspecified tie behaviour of `torch.topk` instead of the "smaller index wins" rule.
