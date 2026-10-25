# In Detail: Knowledge Distillation Temperature Loss

## Origin
Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network* (NIPS 2014 Deep Learning Workshop, arXiv:1503.02531). The soft-target loss is defined as \(T^2\,\mathrm{KL}\big(\mathrm{softmax}(z_t/T)\,\|\,\mathrm{softmax}(z_s/T)\big)\); the \(T^2\) factor exists because differentiating softmax with respect to temperature brings in a \(1/T\), and the gradient magnitudes of the hard and soft losses must remain comparable.

## Core points and pitfalls
1. **The KL direction is teacher‖student**: the teacher distribution is the "reference measure", with \(p_t\) as the weight, written as \(\sum p_t(\log p_t-\log p_s)\). Reversing the direction gives different values.
2. **Divide by T first, then log_softmax**: \(\log\mathrm{softmax}(z/T)\) is numerically stable and avoids an explicit `softmax` followed by `log`.
3. **\(T^2\) before the reduction**: multiply each per-sample KL by \(T^2\) and then apply mean/sum/none, rather than reducing first and multiplying afterwards.
4. **Degenerate anchor**: at \(T=1\), \(1^2\cdot\mathrm{KL}(\mathrm{softmax}(z_t)\|\mathrm{softmax}(z_s))\), used to cross-check against ordinary softmax-KL.
5. **Three reduction modes**: `none` returns an \((N,)\) vector, and an invalid string raises `ValueError`.

## Matching convention
Reference solution: `log_softmax(z/T)→ kl_rows=(p_t*(tt-ts)).sum(-1)→ loss=T*T*kl_rows → reduction`. Grader `_oracle` in pure Python: first hand-compute softmax of each row of \(z/T\) (subtracting the max), then accumulate \(\sum p_t\ln(p_t/p_s)\) per sample by definition, multiply by \(T^2\), and reduce according to reduction; the structure differs but the values are equivalent. Assertions are also set for "T=1 equals ordinary softmax-KL", "identical teacher and student → 0", "example values", and "invalid reduction raises".

## Complexity
\(O(N C)\).
