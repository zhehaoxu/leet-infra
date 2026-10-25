# Solution Notes · P-less Sampling

## Collision probability as the threshold

Draw two tokens independently from distribution $p$; the probability that they are **the same** is $\sum_i p_i^2$ (the collision probability, equal to
the exponential of the Rényi-2 entropy $2^{-H_2}$). P-less's criterion is very plain: **a token that is even less likely than "randomly drawing itself"
is not worth keeping** — i.e. drop the long tail with $p_i < \sum_j p_j^2$.

$$\text{thr} = \sum_i p_i^2,\qquad \text{keep}_i = [\,p_i \ge \text{thr}\,]$$

## Why it is hyperparameter-free

- **Peaked distribution** ($p_{\max}\to 1$): $\sum p^2 \to 1$, the threshold is extremely high, and almost only the argmax remains.
- **Flat distribution** ($V$ equiprobable $1/V$): $\sum p^2 = V\cdot(1/V)^2 = 1/V$, and each $p_i = 1/V$,
  so exactly all are kept.

Thus the same code automatically gives different truncation widths at different confidence levels, without tuning `p` per model as top-p does.

## Key invariants

- **The argmax is always kept**: $\sum_i p_i^2 = \sum_i p_i\cdot p_i \le p_{\max}\sum_i p_i = p_{\max}$,
  so $p_{\max}\ge\text{thr}$ always holds, the `filtered` denominator is always positive, and nothing is ever emptied.
- The threshold uses **probabilities** $p$ (after softmax), not logits; $\sum p^2$ takes values on the probability simplex
  in $\big[1/V,\,1\big]$.

## Implementation notes

`thr = probs.pow(2).sum(-1, keepdim=True)`; `keep = probs >= thr`; `filtered = probs*keep`,
then divide by `filtered.sum(-1, keepdim=True)`. Sampling is the same as in the other problems of this collection: take the first $\ge u$ on the CDF of `filtered` in its original order. Note the use of `>=` (the paper and the official implementation keep entries equal to the threshold).
