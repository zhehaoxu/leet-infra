# MoE Routing: Coefficient of Variation (CV) Auxiliary Loss

## Problem

Another load-balancing penalty proposed by Shazeer in "Outrageously Large Neural Networks": directly penalize the square of the **coefficient of variation** (CV) of **importance** and **load**. CV is dimensionless and inherently invariant to the number of experts and the number of tokens, which makes it convenient to add to the main loss.

$$ \mathrm{CV}(x) = \frac{\mathrm{std}(x)}{\mathrm{mean}(x)}\,,\qquad
\mathcal{L}_{\text{cv}} = \mathrm{CV}(\text{importance})^2 + \mathrm{CV}(\text{load})^2 $$

## Rules

Implement `solve(logits, k)`, where `logits` has shape $(T, E)$:

1. `probs = softmax(logits, -1)`.
2. `importance_e = Σ_t probs[t, e]` (soft, differentiable).
3. `selected = topk(probs, k, -1).indices`; `load_e = count(e ∈ selected)` (hard).
4. $\mathrm{CV}(x) = x.\mathrm{std}(\text{unbiased}=\mathrm{False}) / (x.\mathrm{mean}() + 10^{-6})$.
5. Return the scalar $\mathrm{CV}(\text{importance})^2 + \mathrm{CV}(\text{load})^2$.

> The denominator eps is fixed at $10^{-6}$; std uses the **biased estimate** (dividing by $E$ rather than $E-1$).

## Examples

```
logits = [[2,0,0,0], [0,2,0,0], [0,0,2,0], [0,0,0,2],
          [1,1,0,0], [0,1,1,0]], k=1
cv_aux = 0.122975
```

Extreme contrast: all 4 tokens routed to the same expert → cv_aux ≈ **5.05**.

## Constraints

- CPU float32; ties are broken in favor of the smaller expert index.
