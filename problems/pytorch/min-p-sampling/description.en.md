# Min-p Sampling

## Problem

top-p truncates by "cumulative mass" and still keeps a large pile of noisy tokens when the distribution is flat; top-k, on the other hand, ignores confidence entirely. **Min-p sampling** (Minh & Baker 2023) sets a **floor relative to the highest probability**: keep only tokens whose probability is at least `p · p_max` (`p_max` is the largest probability in the distribution). The more confident the model (`p_max` larger), the higher the floor and the more aggressively it cuts; the more perplexed it is, the more it relaxes automatically. It is especially effective for creative writing at high temperature and for heavily quantized small models, and it is one of the main truncation methods replacing top-k in today's inference engines (llama.cpp defaults to `min-p`).

> To make grading **deterministic**, the uniform random number `u` is passed in as an explicit input, and sampling is implemented with the inverse transform method.

## Rules

- Implement `solve(logits, p, u)`: `logits` has shape `(..., V)`, `p ∈ (0,1)`, and `u` has shape `(...,)` or is a scalar with value in $[0,1)$.
- Steps: 1) `probs = softmax(logits)`; 2) let `p_max = probs.max()`, keep the candidates satisfying `probs >= p · p_max` (the highest-probability one is always kept, so at least 1 remains) and zero out the rest; 3) renormalize the kept part to obtain `filtered`; 4) use `u` to sample by inverse transform over the CDF of `filtered` (in the original order), and return the original index of the selected token.
- Return `(filtered, chosen)`: `filtered` has the same shape as `probs`, and `chosen` has shape `(...,)`.
- Equivalent criterion (softmax is monotone): keep `logit >= logit_max + ln(p)`, which is exactly what llama.cpp does.

## Examples

```
Input: logits = [2, 1, 0.5, -1, -2, -3], p = 0.2, u = 0.9
  probs = [0.600294, 0.220836, 0.133944, 0.029887, 0.010995, 0.004045]
  p_max = 0.600294 → floor = 0.2 · 0.600294 = 0.120059
  keep the first 3 with probs >= 0.120059 (0.029887 < floor is cut)
  normalize (divide by 0.955074) → filtered = [0.628532, 0.231224, 0.140244, 0, 0, 0]
  CDF = [0.628532, 0.859756, 1.0], u = 0.9 → the first ≥0.9 is position 2 → original index 2 is selected
Output: filtered = [0.628532, 0.231224, 0.140244, 0, 0, 0], chosen = 2
```

## Constraints

- float32 input; `p ∈ (0,1)`; `u ∈ [0,1)`
- Each row of `filtered` sums to 1, and cut positions are exactly 0; `chosen` must fall inside the kept set
- Batching over `(..., V)` is supported; `p_max` is taken row-wise along the last dimension
