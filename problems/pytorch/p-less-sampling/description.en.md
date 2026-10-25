# P-less Sampling

## Problem

top-k / top-p / min-p all require a human-chosen hyperparameter (`k`, `p`), which must be retuned whenever the model changes. **P-less**
(Barber et al., arXiv:2509.23234) is a family of **hyperparameter-free** truncation methods: the threshold is computed directly from the distribution itself
and contains no tunable knob. Its criterion comes from the **collision probability** — the probability that two random draws from the distribution $p$ both land on the same
token is $\sum_i p_i^2$ (the Rényi-2 index). P-less specifies that a token is kept only when its own
probability is **no smaller than this collision probability**:

$$\text{keep}_i = \big[\; p_i \ge \textstyle\sum_j p_j^2 \;\big]$$

Intuition: $\sum p^2$ measures how much the distribution "concentrates its mass on a few tokens". The sharper the distribution, the higher the collision probability,
the stricter the threshold, and only the top few are kept; the flatter the distribution, the lower the collision probability, and more tokens are kept.
After truncation, normalize and then sample.

> To guarantee **determinism** for grading, the uniform random number `u` is passed in as an explicit input, and sampling uses the inverse transform method (take the first position in the
> CDF of the kept distribution that is $\ge u$).

## Rules

- Implement `solve(logits, u)`: `logits` has shape `(..., V)`, and `u` is one uniform random number per row with values in $[0,1)$
  (shape `(...,)` or a scalar). **No other hyperparameters**.
- Steps (row by row): ① `probs = softmax(logits)`; ② `thr = (probs**2).sum(-1, keepdim=True)`
  (the collision probability $\sum p^2$); ③ keep `probs >= thr` and set the rest to 0; ④ normalize the kept part → `filtered`
  (same shape as `probs`, each row sums to 1); ⑤ use `u` to inverse-transform sample on the CDF of `filtered` (in the original order) to get `chosen`.
- Return `(filtered, chosen)`: `filtered` has the same shape as `probs`, and `chosen` has shape `(...,)`.

## Examples

```
Input: logits = [1, 0.8, 0.6, 0.4, 0.2, 0, -0.2], u = 0.7
  probs(softmax) = [0.240601, 0.196987, 0.161279, 0.132044, 0.108109, 0.088512, 0.072468]
  thr = Σ p² = 0.164913
  keep p >= 0.164913 → indices {0, 1} (0.2406, 0.1970; 0.1613 < thr is dropped)
  normalize (divide by 0.437588) → filtered = [0.549834, 0.450166, 0, 0, 0, 0, 0]
  CDF = [0.549834, 1.0], u = 0.7 → the first ≥0.7 is position 1 → original index 1 is chosen
Output: filtered = [0.549834, 0.450166, 0, 0, 0, 0, 0], chosen = 1
```

## Constraints

- float32 input; `u ∈ [0,1)`; no policy hyperparameters
- Each row of `filtered` sums to 1, dropped positions are exactly 0; `chosen` always falls in the kept set
- Supports batched `(..., V)`; the argmax is always kept (its $p_{\max}\ge\sum p^2$ always holds)
