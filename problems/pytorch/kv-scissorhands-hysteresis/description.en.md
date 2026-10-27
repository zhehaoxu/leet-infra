# KV Cache: ScissorHands Temporal Hysteresis Eviction

## Problem

Evicting KV cache entries by **single-frame** attention scores has a fatal weakness: the score itself is noisy, so a critical token may be judged unimportant after one transient attention shift, become important again on the next frame, and by then it has already been permanently deleted. ScissorHands (Liu et al., NeurIPS 2023, arXiv:2305.17118) names this the **persistence of importance** hypothesis: token importance changes slowly, so the **eviction decision should also carry memory**.

Its mechanism is a per-token **consecutive low-score counter**: a token is evicted only when its importance stays below the threshold \(\tau\) for \(m\) **consecutive** frames; any single frame above the threshold resets the counter. This is the classical control-theoretic idea of **hysteresis**—a patience window that filters single-frame jitter, at the cost of reacting \(m-1\) frames late to genuine staleness.

Formally, at step \(s\) for token \(j\):

\[
c_s[j]=
\begin{cases}
c_{s-1}[j]+1, & \text{score}_s[j]<\tau\\[2pt]
0, & \text{score}_s[j]\ge\tau
\end{cases}
\qquad
\text{evict}_s[j]=1 \iff \big(c_s[j]\ge m\big)\ \wedge\ \text{not yet evicted}
\]

Eviction is **irreversible**: once evicted at step \(s\), the token is dead in every later step's mask (in a real cache it no longer exists, so later scores must not be able to rescue it).

## Task

Implement `solve(scores, tau, patience)`:

- Input: `scores` (float32 `(S, L)`, per-step importance for `L` tokens over `S` steps); `tau` (threshold scalar); `patience` (integer \(m\ge 1\)).
- Output `(evict_step, keep)`: `evict_step` is `(L,)` int64 with the step index of the first eviction, `-1` if never evicted; `keep` is `(S, L)` bool where `keep[s]` is the survival mask **after** processing evictions at step `s`.
- A low score is **strictly less** than the threshold: `score == tau` is not low and resets the counter.
- Raise `ValueError` when `patience < 1`.

## Example

```
scores = [[0.9, 0.1, 0.6, 0.2],
          [0.8, 0.2, 0.4, 0.3],
          [0.7, 0.3, 0.6, 0.9],
          [0.6, 0.4, 0.5, 0.8]]      tau = 0.5, m = 2

token0: 0.9,0.8,0.7,0.6  -> never low                 -> evict_step = -1
token1: 0.1,0.2,...      -> low on steps 0 and 1      -> evict_step = 1
token2: 0.6,0.4,0.6,...  -> low at step 1, then recovers -> evict_step = -1
token3: 0.2,0.3,...      -> low on steps 0 and 1      -> evict_step = 1

evict_step = [-1, 1, -1, 1]
keep[1]    = [True, False, True, False]
```

## Constraints

- Once evicted, a token's counter stops updating and it can never come back: `keep[s][j]` is 0 for all `s >= evict_step[j]`.
- `patience = 1` degenerates to plain single-frame threshold eviction (`score < tau` evicts immediately).
- Counters are per token: low-score runs of different tokens never interact.
- Each step must update the counter **before** testing it; reversing the order shifts every `evict_step` by one.
