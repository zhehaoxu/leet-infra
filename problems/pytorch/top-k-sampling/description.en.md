# Top-k Sampling

## Problem

In the vocabulary distribution produced by a model, the long-tail tokens each have tiny probability
but there are a huge number of them, so sampling from the full distribution easily spits out nonsense.
**Top-k sampling** (Fan et al. 2018) is the crudest truncation method: keep only the `k` tokens with the
highest probability (equivalently, the highest logit), renormalize them into a new distribution, and sample
from it. It bluntly cuts every candidate after the `k`-th one regardless of how confident the model is —
exactly what top-p / min-p later tried to improve on. It is still one stage of the default pipeline in
almost every inference engine (including llama.cpp's `top-k`).

> To keep judging **deterministic**, the uniform random number `u` is passed in explicitly, and sampling
> uses the inverse-transform method (take the first position on the CDF of the kept distribution with
> value $\ge u$).

## Rules

- Implement `solve(logits, k, u)`: `logits` has shape `(..., V)`, `k` is an integer (`k≥V` keeps everything),
  and `u` is a uniform random number in $[0,1)$ per row (shape `(...,)` or a scalar).
- Steps: ① `probs = softmax(logits)`; ② keep the `k` largest logits (ties broken by original order, earliest
  first) and zero out the rest; ③ renormalize the kept part to obtain `filtered` (same shape as `probs`,
  each row sums to 1); ④ use `u` to inverse-transform sample on the CDF of `filtered` (in original order),
  and return the original index of the selected token.
- Return `(filtered, chosen)`: `filtered` has the same shape as `probs`, `chosen` has shape `(...,)`.

## Example

```
Input: logits = [2, 1, 0.5, -1, -2, -3], k = 2, u = 0.9
  probs(softmax) = [0.600294, 0.220836, 0.133944, 0.029887, 0.010995, 0.004045]
  keep top-2 = [0.600294, 0.220836], zero the rest
  normalize (divide by 0.821130) → filtered = [0.731059, 0.268941, 0, 0, 0, 0]
  CDF = [0.731059, 1.0], u = 0.9 → first ≥0.9 is position 1 → selects original index 1
Output: filtered = [0.731059, 0.268941, 0, 0, 0, 0], chosen = 1
```

## Constraints

- float32 input; `k ≥ 1`; `u ∈ [0,1)`
- Every row of `filtered` sums to 1, and non-top-k positions are exactly 0; `chosen` always lies in the kept set
- Support batching over `(..., V)`; tie-breaking among equal logits is stable by original index order
