# In Detail: SigLIP Sigmoid Contrastive Loss

## Source
Zhai et al., *Sigmoid Loss for Language-Image Pre-Training* (arXiv:2303.15343), Eq. (2): \(\mathcal{L}=-\frac1B\sum_i\sum_j\log\sigma(y_{ij}(t\,s_{ij}+b))\), with \(y=+1\) on the diagonal and \(-1\) elsewhere, and \(t,b\) being the learnable scale/bias. Each pair is independent, with no softmax global normalization.

## Core points and common mistakes
1. **It is sigmoid, not softmax**: each pair is an independent binary classification, so there is no "row-wise normalization" and no need to gather the whole batch of similarities across devices.
2. **The denominator is \(B\), not \(B^2\)**: sum over all \(B^2\) pairs, then divide by the batch size \(B\) (as in the paper's pseudocode); averaging over \(B^2\) instead differs by a factor of \(B\).
3. **Labels are ±1**: \(y_{ij}=2\mathbb{1}[i=j]-1\), which flips the \(z\) of negative pairs to "larger is better" as well. An equivalent formulation is \(y\odot z\) followed by `logsigmoid`.
4. **The affine transform comes before the sigmoid**: \(z=t\,s+b\); do not drop the bias, and do not multiply outside the loss.

## Cross-check convention
The reference solution is `-(F.logsigmoid((2*eye-1)*(scale*sim+bias)).sum())/B`. The grading `_oracle` uses a Python double loop accumulating the stable form `log σ(m)=-log1p(exp(-m))` per pair and divides by B (different structure, numerically equivalent), with `abs(got-exp)<1e-5`. It also asserts "very large diagonal and very small off-diagonal → loss approaches 0", "the closed form for scale=0 with a fixed bias", and "square shape / scalar return".

## Complexity
\(O(B^2)\) over pairs.
