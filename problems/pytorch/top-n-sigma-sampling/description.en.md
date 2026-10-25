# Top-n-sigma Sampling

## Problem

top-k and top-p both require a human to pick a knob — "how many to keep / how much mass to keep" —
independent of the shape of the distribution itself.
**Top-n-sigma** (llama.cpp's `top-n-sigma`) takes a different approach: it ties the truncation threshold
to the **statistical spread** of the distribution. It computes the mean $\mu$ and the (population)
standard deviation $\sigma$ of the whole row of logits, and keeps only the candidates with
$\text{logit} \ge \max - n\sigma$. The flatter the distribution (the larger $\sigma$), the more is kept;
the more peaked the distribution (the smaller $\sigma$), the harder the cut — the threshold adapts to
confidence without any fixed `k`.

> To keep judging **deterministic**, the uniform random number `u` is passed in explicitly, and sampling
> uses the inverse-transform method (take the first position on the CDF of the kept distribution with
> value $\ge u$).

## Rules

- Implement `solve(logits, n, u)`: `logits` has shape `(..., V)`, `n` is a positive float (the number of
  sigmas), and `u` is a uniform random number in $[0,1)$ per row (shape `(...,)` or a scalar).
- Steps (row by row along the last dim): ① `probs = softmax(logits)`; ② let `max = logits.max()`,
  `mean = logits.mean()`, `std = sqrt(mean((logits - mean)^2))` (**population** standard deviation,
  divided by `V`); ③ keep candidates with `logit >= max - n*std` and zero out the rest; ④ renormalize the
  kept part → `filtered` (same shape as `probs`, each row sums to 1); ⑤ use `u` to inverse-transform
  sample on the CDF of `filtered` (in original order) to obtain `chosen`.
- Return `(filtered, chosen)`: `filtered` has the same shape as `probs`, `chosen` has shape `(...,)`.

## Example

```
Input: logits = [3, 2.5, 0, -0.2, -0.5, -4], n = 1.0, u = 0.8
  probs(softmax) = [0.578640, 0.350963, 0.028809, 0.023587, 0.017473, 0.000528]
  mean = 0.133333, std = 2.288862, max = 3.0
  threshold cut = max - n*std = 3.0 - 2.288862 = 0.711138
  keep logit >= 0.711138 → indices {0, 1} (values 3.0, 2.5)
  normalize → filtered = [0.622459, 0.377541, 0, 0, 0, 0]
  CDF = [0.622459, 1.0], u = 0.8 → first ≥0.8 is position 1 → selects original index 1
Output: filtered = [0.622459, 0.377541, 0, 0, 0, 0], chosen = 1
```

## Constraints

- float32 input; `n > 0`; `u ∈ [0,1)`
- `std` is the population standard deviation (divide by the element count `V`, not `V-1`)
- Every row of `filtered` sums to 1, discarded positions are exactly 0; `chosen` always lies in the kept set
- Support batching over `(..., V)`; `mean/std/max` are computed independently per row
