# Sparsification: Soft-Thresholding and One ISTA Step

## Problem

Minimising the L1-regularised least squares \(\min_x \tfrac12\|Wx-y\|_2^2+\lambda\|x\|_1\) shows up everywhere from sparse coding to learned pruning masks. Its proximal-gradient method (ISTA, iterative shrinkage-thresholding) does only two things per step:

1. take a gradient step on the data term;
2. apply **soft-thresholding**, the proximal operator of the L1 norm.

Soft-thresholding is

$$
\mathcal{S}_\lambda(z)=\operatorname{sign}(z)\cdot\max(|z|-\lambda,\,0)
$$

It shrinks every component **towards zero by a fixed amount \(\lambda\)** and clamps anything smaller than \(\lambda\) to exactly zero — that is the geometric origin of L1 sparsity: the whole neighbourhood of the threshold is flattened. Contrast this with the ubiquitous top-k hard pruning (`sparsity-magnitude-prune`): hard pruning keeps a fixed **count** regardless of magnitudes, whereas soft-thresholding keeps everything "worth keeping", with the count determined jointly by \(\lambda\) and the weight distribution — and it also changes the surviving values, which is why it is (sub)differentiable everywhere except at 0.

## Rules

Given `x` (float32, \((n,)\)), `W` (float32, \((m,n)\)), `y` (float32, \((m,)\)), scalar `lam` (\(\lambda\ge0\)) and step size `eta` (\(\eta\ge0\)):

1. residual \(r=Wx-y\);
2. gradient \(g=W^\top r\) of the data term \(\tfrac12\|Wx-y\|^2\) with respect to \(x\);
3. gradient step \(z=x-\eta g\);
4. elementwise `x_new = sign(z) * max(|z| - lam, 0)`;
5. `support` = indices with \(|x\_new|>0\), in **ascending** order.

Return `(x_new, support)`: the first is float32 \((n,)\), the second int64 1-D. `torch.nn.functional.softshrink` and other off-the-shelf proximal operators are forbidden.

## Examples

```
x = [1, -2, 0.5]
W = [[1, 0, 0],
     [0, 1, 1]]
y = [1, 1]
lam = 0.2, eta = 0.5

r = Wx - y  = [1 - 1, (-2 + 0.5) - 1] = [0, -2.5]
g = Wᵀ r    = [0, -2.5, -2.5]
z = x - 0.5*g = [1, -0.75, 1.75]

x_new = [1-0.2, -(0.75-0.2), 1.75-0.2] = [0.8, -0.55, 1.55]
support = [0, 1, 2]
```

## Constraints

- CPU float32; `lam` and `eta` are non-negative scalars.
- Components with \(|z|\) **exactly equal** to `lam` must become 0 (the keep condition is strictly greater); `support` must not contain them.
- `lam=0` must degenerate exactly to the gradient step \(x-\eta g\); `eta=0` performs shrinkage only.
- Shrinkage preserves sign and only reduces magnitude: kept components satisfy \(|x\_new|=|z|-\lambda\).
