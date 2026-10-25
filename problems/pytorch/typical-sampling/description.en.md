# Typical / Locally Typical Sampling (ε-sampling)

## Problem

top-k / top-p all assume "higher probability deserves to be kept". **Locally typical sampling**
(Tavakoli et al. 2020, called `typical` in llama.cpp) goes the other way: it keeps tokens whose
**information content is typical** — that is, those whose self-information (surprisal) $-\ln p$ is close to
the entropy of the whole distribution $H=\sum_i -p_i\ln p_i$. The intuition is that only "expected" words
should be sampled; it removes both surprising long-tail tokens (surprisal much larger than $H$) and
**over-confident** argmax tokens (surprisal much smaller than $H$) — the latter being the key quirk that
distinguishes it from every truncation method.

> To keep judging **deterministic**, the uniform random number `u` is passed in explicitly, and sampling
> uses the inverse-transform method.

## Rules

- Implement `solve(logits, tau, u)`: `logits` has shape `(..., V)`, `tau ∈ (0,1)` is the cumulative mass
  threshold, and `u` has shape `(...,)` or is a scalar with value in $[0,1)$.
- Steps: ① `probs = softmax(logits)`; ② entropy $H=\sum_i -p_i\ln p_i$; ③ each token's typicality deviation
  $d_i = |(-\ln p_i) - H|$; ④ sort tokens by $d_i$ **ascending** (ties by ascending original index),
  accumulate probability in that order, and keep tokens up to and including the one where the cumulative sum
  **first exceeds tau** (keep at least 1); ⑤ renormalize the kept set in original order to obtain `filtered`;
  ⑥ use `u` to inverse-transform sample on the CDF of `filtered` (in original order), returning the original
  index of the selected token.
- Return `(filtered, chosen)`.

## Example

```
Input: logits = [3, 2.95, -5, -5, -5, -5, -5, -5, -5, -5], tau = 0.4, u = 0.3
  probs ≈ [0.511793, 0.486833, 0.000172 ×8]
  entropy H = 0.705164
  surprisal: token0 = 0.669834 → d0 = |0.669834-0.705164| = 0.035330
             token1 = 0.719834 → d1 = |0.719834-0.705164| = 0.014670
             long tail surprisal ≈ 8.67 → d ≈ 7.96
  ascending by d: token1(0.0147) < token0(0.0353) < long tail
  accumulate: take token1 first, cum = 0.486833 > tau=0.4 → stop, keep only {token1}
  ⚠️ token0 (the argmax) is discarded — the signature behavior of typical sampling
  normalize → filtered = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], chosen = 1
Output: filtered = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], chosen = 1
```

## Constraints

- float32 input; `tau ∈ (0,1)`; `u ∈ [0,1)`
- Every row of `filtered` sums to 1, discarded positions are exactly 0; `chosen` always lies in the kept set
- Support batching over `(..., V)`; logarithms are $\ln$, and $p=0$ is floored at `1e-30`; ties are broken
  stably by original index order
- Unlike top-k/top-p/min-p: this method **may discard the argmax**
