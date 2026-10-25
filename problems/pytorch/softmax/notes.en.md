# Softmax Explained

## What it actually computes

Softmax maps a whole row of real numbers $z$ into a probability distribution, preserving order and summing to 1:

$$ \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}} $$

It is the combination of "exponential family + normalization", and the shared kernel of cross entropy, attention weights and temperature sampling. Understanding softmax is not about the formula but about why it **must subtract the maximum**.

## Numerical stability: subtracting the max is not an optimization, it is a necessity

In `float32`, $e^{x}$ overflows to `inf` for $x \gtrsim 88$ and underflows to 0 for $x \lesssim -104$. If you sum `exp` directly:

- large logits (e.g. the last layer of a classifier, or attention $QK^\top$) → `inf/inf = NaN`;
- all-negative logits → the denominator tends to 0.

Use translation invariance: $\text{softmax}(z) = \text{softmax}(z - c)$ with $c = \max_i z_i$. Then every exponent argument lies in $(-\infty, 0]$, so $e^{\cdot} \in (0,1]$ and overflow is impossible. The denominator always contains the $\max$ term, which equals $1$, so it can never be 0.

> After subtracting the max, extremely negative terms may still underflow to 0 — that is harmless, since they correspond to probabilities ~0 and do not affect the sum or the normalization accuracy.

## Relation to temperature and Top-k/Top-p

Temperature scaling divides by $T$ before the softmax: $\text{softmax}(z/T)$. As $T\to 0$ it approaches one-hot(argmax); as $T\to\infty$ it approaches the uniform distribution. The "temperature scaling and Top-p" problem in this collection extends exactly this chain.

## Gradient (for understanding; implementation is not required)

With $p = \text{softmax}(z)$, the Jacobian is

$$ \frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j) $$

Combined with cross entropy the gradient becomes beautifully simple: $\partial \mathcal{L}/\partial z = p - y$ (prediction minus one-hot label), which is the shape of the backward pass at the last layer of every classification head.

## Common pitfalls

1. **Wrong dimension**: normalize rows with `dim=-1`; in batched attention it is along the key axis, so do not mistakenly use `dim=0`.
2. **All values in a row identical**: for $z=[a,a,a]$, no matter how large $a$ is, subtracting the max must give the uniform $[1/3,\dots]$; without it you may get `inf/inf`.
3. **Rows of all -inf** (masked out): subtracting the max yields `NaN` and needs special handling (torch attention conventionally outputs 0). Not part of this level, but watch out in the linked problems.
