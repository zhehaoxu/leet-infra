# In Depth: SimPO Length-Normalized Preference Loss

## Source
Meng et al., *SimPO: Simple Preference Optimization with a Reference-Free Reward* (arXiv:2405.14734). The reward is the average log-probability \(r=\frac1{|y|}\log\pi_\theta(y|x)\) (no reference model, length-normalized), and the loss is \(-\log\sigma(\beta(r_w-r_l)-\gamma)\), where \(\gamma\) is the target margin (the paper folds it in equivalently as \(-\beta\gamma\); this problem implements the formula as stated).

## Core idea and pitfalls
1. **Divide by length before subtracting**: \(r_w=\text{logp}_w/\text{len}_w\) — it is the difference of "average log-probabilities", not a direct subtraction of sequence log-probabilities. Length normalization is the soul of SimPO.
2. **Subtract γ inside the margin**: \(\beta(r_w-r_l)-\gamma\); the larger γ is, the bigger the gap must be before the loss drops. Forgetting to subtract γ degenerates into a plain sigmoid-DPO form.
3. **Numerically stable logsigmoid**: use `F.logsigmoid`, or compute `log(1+e^{m})` from `-m`, to avoid overflow when exponentiating large values.
4. Average the loss over the batch, then negate it.

## Cross-check methodology
The reference solution is `(-F.logsigmoid(beta*(lw/nw - ll/nl) - gamma)).mean()`. The judging `_oracle` recomputes element-wise with the stable form `logsigmoid(m)=-log1p(exp(-m))` built on `math.log1p(exp(...))` (different structure, numerically equivalent), with `allclose(atol=1e-5,rtol=1e-5)`. There are also assertions for "γ=0 with unnormalized logp as r matches the naive formula" and "length normalization matters".

## Complexity
\(O(N)\) element-wise.
