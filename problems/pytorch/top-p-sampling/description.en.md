# Top-p / Nucleus Sampling

## Problem

Temperature / top-k sampling is either too diffuse or truncates too bluntly. **Nucleus sampling**
(Holtzman et al. 2021) dynamically selects "the smallest set of tokens with the highest probabilities
whose cumulative mass first reaches `p`" (the nucleus), and samples only after renormalizing inside the
nucleus — the nucleus is small when the distribution is peaked and large when it is flat, balancing
diversity and plausibility. It is one of the default decoding strategies in nearly every LLM inference
engine.

> To keep judging **deterministic**, the uniform random number `u` is passed in explicitly, and sampling
> is done with the inverse-transform method (CDF).

## Rules

- Implement `solve(logits, p, u)`: `logits` has shape `(..., K)`, `p∈(0,1]`, and `u` is a uniform random
  number in $[0,1)$ per row (shape `(...,)` or a scalar).
- Steps: ① `probs=softmax(logits)`; ② sort descending and take the cumulative sum `cumsum`; ③ keep elements
  up to the one where the cumulative sum **first reaches ≥ p** (always keep at least 1; the test is
  `cumsum - sorted_prob < p`); ④ normalize inside the nucleus and scatter back to the original order to get
  `filtered` (zero outside the nucleus); ⑤ use `u` to inverse-transform sample on the in-nucleus CDF and
  return the original index of the selected token.
- Return `(filtered, chosen)`: `filtered` has the same shape as `probs` and each row sums to 1; `chosen`
  has shape `(...,)`.

## Example

```
Input: logits = [2, 1, 0.5, -1, -2]
  probs = [0.6027, 0.2217, 0.1345, 0.0300, 0.0110]
  descending cumulative sum = [0.6027, 0.8245, 0.9590, 0.9890, 1.0]
  p = 0.8 → nucleus = first 2 (0.6027 and 0.2217; the sum reaches 0.8245 ≥ 0.8 only at the second)
  normalize inside nucleus = [0.7311, 0.2689] → scatter back to original order
  u = 0.05, in-nucleus CDF=[0.7311,1.0], first ≥0.05 is position 0 → selects original index 0
Output: filtered = [0.7311, 0.2689, 0, 0, 0], chosen = 0
```

## Constraints

- float32 input; `p∈(0,1]`, `u∈[0,1)`
- Every row of `filtered` sums to 1 and is 0 outside the nucleus; `chosen` always lies in the nucleus
