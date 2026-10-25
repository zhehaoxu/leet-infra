# Explained: Focal Loss

## Source
Lin, Goyal, Girshick, He, Dollar, *Focal Loss for Dense Object Detection* (ICCV 2017, arXiv:1708.02002, RetinaNet). The binary definition is \(\text{FL}(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t\), \(p_t=y\,p+(1-y)(1-p)\). This problem takes its multi-class (softmax) form; the relation to cross-entropy is that CE is multiplied by the modulating factor \((1-p_t)^\gamma\).

## Core Points and Common Mistakes
1. **The modulating factor acts in probability space, not logit space**: first softmax to get \(p_t\), then compute \((1-p_t)^\gamma\).
2. **Use `log_softmax` + gather for \(\log p_t\)**: directly computing `log(softmax)` easily overflows with extreme logits; `log_softmax` is numerically stable.
3. **Degenerates to cross-entropy**: with \(\gamma=0,\alpha=\text{None}\), \(-(1-p_t)^0\log p_t=-\log p_t\) = the per-sample NLL of standard CE, so it must exactly match `F.cross_entropy(reduction='none')`—this problem uses it as the cross-check anchor.
4. **alpha is a scalar weight**: this simplified version multiplies every sample by the same `alpha` (corresponding to the positive-class weight \(\alpha_t\) in the original paper), not a per-class vector.

## Cross-Checking Criteria
Reference solution `log_softmax→gather→loss=-(1-pt)^gamma*log_pt`, reduced according to `reduction`, with an invalid reduction raising `ValueError`. The judge's `_oracle` is pure Python: it hand-computes each row's softmax (subtracting the max), takes the true-class probability, and accumulates by definition (structurally different, numerically equivalent). There are also assertions for "γ=0 agrees with F.cross_entropy", "the larger γ, the higher the share of hard samples (for the same batch, the mean decreases monotonically relative to CE)", "alpha scales linearly", and "reduction none/sum/mean agree".

## Complexity
\(O(N C)\).
