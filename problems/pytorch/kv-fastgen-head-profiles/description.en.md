# KV Cache: FastGen Head Profiling and Per-Type Retention Budgets

## Problem

The key insight of FastGen (Ge et al., ICLR 2024, arXiv:2310.01801) is that **different attention heads need structurally different caches**, so a single top-k rule for all of them both wastes memory and hurts quality. The paper profiles each head's attention pattern into a few classes and pairs each class with its own compression strategy:

- **specialization heads**: attention concentrates on a few fixed tokens (delimiters, punctuation, template slots) and the profile is a spike—keep the **highest-scoring** positions;
- **retrieval heads**: attention spreads over the whole prompt (they need to look back at the source text) and the profile is flat—keep **more** positions, i.e. the least compression;
- **streaming heads**: attention concentrates on the most recent window (local syntax) and the profile is right-skewed—**sink + last window** is enough, no scoring needed.

This problem specifies a **fully deterministic toy version of the decision** (the real FastGen runs a DP search over candidate policies at calibration time, which is a one-off training cost and unsuitable for a single judge problem). Given each head's accumulated attention row \(s_h\in\mathbb R^{L}\) (non-negative), compute two profile statistics:

\[
R_h=\frac{\sum_{j\ge L-w}s_h[j]}{\sum_j s_h[j]},\qquad
C_h=\frac{\max_j s_h[j]}{\sum_j s_h[j]}
\]

where \(R_h\) is the **recency mass ratio** (fraction of total mass in the last \(w\) columns) and \(C_h\) is the **concentration** (fraction in the single largest column). Classification follows a fixed priority:

\[
\text{type}_h=
\begin{cases}
\text{streaming (0)}, & R_h\ge r_{\text{hi}}\\
\text{specialized (1)}, & R_h<r_{\text{hi}}\ \wedge\ C_h\ge c_{\text{hi}}\\
\text{retrieval (2)}, & \text{otherwise}
\end{cases}
\]

Retained sets per type (\(n_s\) is the sink count):

| Type | Retained set | Budget |
| --- | --- | --- |
| streaming (0) | first \(n_s\) ∪ last \(w\) | positional, no scoring |
| specialized (1) | first \(n_s\) ∪ top-\(k_s\) of the rest | small \(k_s\) |
| retrieval (2) | first \(n_s\) ∪ top-\(k_r\) of the rest | large \(k_r\) |

This is the "**unequal budget per head**" idea: within one layer a streaming head may keep 5 positions while a retrieval head keeps 20, instead of both keeping 10.

## Task

Implement `solve(scores, n_sink, window, k_spec, k_retr, recency_ratio, concentration)`:

- Input: `scores` (float32 `(H, L)`, non-negative accumulated attention); `n_sink`; `window` (\(w\in[1,L]\)); `k_spec`, `k_retr` (non-negative; larger than the candidate count is allowed and then everything is taken); `recency_ratio` (\(r_{\text{hi}}\)) and `concentration` (\(c_{\text{hi}}\)), both in \([0,1]\).
- Output `(head_type, keep, kept)`: `head_type` is `(H,)` int64 (0/1/2); `keep` is `(H, L)` bool; `kept` is `(H,)` int64 (per-head retained count).
- Ties in top-k go to the **smaller index**; sink positions are excluded from the top-k candidates (they are already kept unconditionally).
- A head with total mass 0 has \(R_h=C_h=0\) and falls into retrieval.
- Raise `ValueError` for out-of-range arguments (`window ∉ [1,L]`, `n_sink ∉ [0,L]`, `k < 0`, thresholds outside [0,1], negative entries in `scores`).

## Example

```
L = 8, n_sink = 1, window = 2, k_spec = 2, k_retr = 4, r_hi = 0.5, c_hi = 0.25

h0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0, 3.0]   R = 5.0/5.6 = 0.89 -> streaming
     keep = [T, F, F, F, F, F, T, T]                     (sink 0 + last window 6,7 = 3)
h1 = [3.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]   R = 0.09, C = 0.68 -> specialized
     keep = [T, T, T, F, F, F, F, F]                     (sink 0 + top2 -> 1,2, tie -> smaller)
h2 = [1,1,1,1,1,1,1,1]                          R = 0.25, C = 0.125 -> retrieval
     keep = [T, T, T, T, T, F, F, F]                     (sink 0 + top4 -> 1..4)

head_type = [0, 1, 2]
kept      = [3, 3, 5]
```

## Constraints

- Classification tests streaming → specialized → retrieval **in that order**: when both `R` and `C` qualify, the head is streaming.
- Streaming heads **ignore scores entirely**: the last window is kept even if it is all zeros, and a huge spike in the middle is still dropped (this is the essential difference between a positional and a scoring policy).
- `k_spec` and `k_retr` are intentionally different scales: retrieval heads keeping more is the design intent, so do not collapse them into one top-k.
- Top-k keeps `min(k, L-n_sink)` positions and never pads when candidates run out.
