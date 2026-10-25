# P-less-norm Sampling

## Problem

P-less-norm is a **relaxed version** of [P-less](#): also hyperparameter-free, also using the collision probability $\sum p^2$ as the criterion,
but it **normalizes the threshold by the vocabulary size**, so the threshold is lower than the original P-less and thus keeps more candidates and favors diversity
(the paper recommends it for scenarios where "diversity matters more than coherence").

The original P-less threshold is $\text{thr}=\sum_i p_i^2$. P-less-norm rewrites it as — let $v=V$ be the vocabulary size:

$$\text{thr} = \frac{v\,\sum_i p_i^2 - 1}{v - 1}$$

Since $\sum p^2\in[1/v,\,1]$, this threshold affinely maps $[1/v,1]$ back to $[0,1]$: for a uniform distribution the threshold drops to $0$
(keep everything), and for a peaked distribution it approaches $1$. Compared with the original, tokens in the middle band are more easily kept. The criterion is likewise
$p_i \ge \text{thr}$.

> To guarantee **determinism** for grading, the uniform random number `u` is passed in as an explicit input, and sampling uses the inverse transform method (take the first position in the
> CDF of the kept distribution that is $\ge u$).

## Rules

- Implement `solve(logits, u)`: `logits` has shape `(..., V)`, and `u` is one uniform random number per row with values in $[0,1)$
  (shape `(...,)` or a scalar). **No other hyperparameters**.
- Steps (row by row): ① `probs = softmax(logits)`; ② let `v = probs.shape[-1]`,
  `thr = (v * (probs**2).sum(-1, keepdim=True) - 1) / (v - 1)`; ③ keep `probs >= thr` and
  set the rest to 0; ④ normalize → `filtered` (same shape as `probs`, each row sums to 1);
  ⑤ use `u` to inverse-transform sample on the CDF of `filtered` (in the original order) to get `chosen`.
- Return `(filtered, chosen)`: `filtered` has the same shape as `probs`, and `chosen` has shape `(...,)`.

## Examples

```
Input: logits = [1, 0.8, 0.6, 0.4, 0.2, 0, -0.2], u = 0.5
  probs(softmax) = [0.240601, 0.196987, 0.161279, 0.132044, 0.108109, 0.088512, 0.072468]
  v = 7, Σp² = 0.164913
  thr = (7 * 0.164913 - 1) / 6 = 0.025732
  keep p >= 0.025732 → all 7 satisfy it (the smallest, 0.0725, is also ≥ the threshold)
  normalize → filtered = probs = [0.240601, 0.196987, 0.161279, 0.132044, 0.108109, 0.088512, 0.072468]
  CDF = [0.2406, 0.4376, 0.5989, ...], u = 0.5 → the first ≥0.5 is position 2 → original index 2 is chosen
Output: filtered = probs, chosen = 2
```

## Constraints

- float32 input; `u ∈ [0,1)`; no policy hyperparameters
- `V ≥ 2` (the denominator `v-1` must be positive)
- Each row of `filtered` sums to 1, dropped positions are exactly 0; `chosen` always falls in the kept set
- Supports batched `(..., V)`; the argmax is always kept (thr ≤ p_max always holds)
