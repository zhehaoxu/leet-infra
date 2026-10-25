# QK-Norm Attention (Normalized QK Attention Score)

## Problem

QK-Norm applies **RMSNorm** to the Query and the Key separately before the dot product, fixing the scale of each vector, and then scales by $1/\sqrt{d}$. This keeps the attention logits bounded so they do not explode as vector norms grow (used by ViT-22B, Chameleon, etc.).

$$ \hat{q} = \frac{q}{\sqrt{\mean(q^2)+\epsilon}}\cdot w_q,\qquad
\hat{k} = \frac{k}{\sqrt{\mean(k^2)+\epsilon}}\cdot w_k,\qquad
\text{score} = \frac{\hat{q}\,\hat{k}^{\top}}{\sqrt{d}} $$

## Rules

- Implement `solve(q, k, weight_q, weight_k, eps=1e-6)` with `q/k` of shape `(..., L, d)`, returning a score matrix of shape `(..., L_q, L_k)`.
- RMSNorm is applied along the **last dimension** and **does not subtract the mean** (unlike LayerNorm): divide by $\sqrt{\mean(x^2)+\epsilon}$, with $\epsilon$ **inside** the square root. `weight_q` and `weight_k` broadcast over the last dimension and may be asymmetric.
- Normalize first, then take the dot product and divide by $\sqrt{d}$.
- **Scale invariance**: multiplying both $q$ and $k$ by any positive constant leaves the scores unchanged (the headline property of this problem, and it is checked by the tests).
- With all-ones weights, each RMS-normalized vector has row norm $\sqrt{d}$, so by Cauchy-Schwarz $|\text{score}| \le \sqrt{d}$.

## Examples

```
Input: q=[[[1,1]]], k=[[[1,1]]], weight_q=weight_k=[1,1], eps=1e-6, d=2
   q̂ = (1,1)/sqrt(mean([1,1]^2)+eps) = (1,1)/1.0 ≈ (1,1)   # mean(q²)=1
   score = (1*1+1*1)/sqrt(2) = 2/√2
Output: [[[1.4142122]]]   # ≈ √2
```

## Constraints

- float32/float64 inputs
- Must match the hand-derived `rmsnorm(q,wq) @ rmsnorm(k,wk)^T / sqrt(d)`
