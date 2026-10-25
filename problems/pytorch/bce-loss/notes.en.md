# Binary Cross-Entropy Loss Explained

## Definition

The negative log-likelihood of a Bernoulli distribution. Ground truth $y\in\{0,1\}$, predicted probability $p\in(0,1)$:

$$ \mathcal L = -\big[y\log p + (1-y)\log(1-p)\big] $$

For $y=1$ this degenerates to $-\log p$ (we want $p\to1$), and for $y=0$ to $-\log(1-p)$ (we want $p\to0$).

## Gradient

$$ \frac{\partial\mathcal L}{\partial p}=\frac{p-y}{p(1-p)} $$

When combined with the sigmoid output, the chain rule cancels out to $\partial\mathcal L/\partial z = p - y$ ($z$ being the logits) — as clean as softmax+CE. This is exactly why **`BCEWithLogitsLoss` is recommended** (see below).

## Probability version vs logits version (the key difference)

| | Input | Numerically stable approach | PyTorch API |
|--|------|--------------|-------------|
| **BCE** (this problem) | probability $p\in(0,1)$ (already through sigmoid) | clamp $p$ to $[\epsilon, 1-\epsilon]$ before $\log$ | `F.binary_cross_entropy` |
| **BCEWithLogits** | logits $z\in\mathbb R$ (not through sigmoid) | fuse sigmoid and log into a softplus form | `F.binary_cross_entropy_with_logits` |

Production code prefers **WithLogits**: it fuses `sigmoid` and `log`, fundamentally avoiding $\log 0=-\infty$ when $p$ saturates to 0/1. This problem practices the probability version, and the trap is exactly that you **must clamp**: `p.clamp(min=eps, max=1-eps)` before the log; the last-bit floating-point difference is within tolerance.

## Contract

- `mean` divides by the total number of elements `numel` (batch × classes), which suits multi-label; `sum` and `none` are the same as for MSE.
- `pos_weight` (class-imbalance weighting) is a common BCE extension; this problem does not implement it.
- `target` may be soft labels (continuous values in $[0,1]$) and the formula still holds — this is how label smoothing is written for multi-label tasks.

## Related

sigmoid provides the probability output; cross-entropy is the multi-class version (one-hot vs multiple Bernoulli); GAN discriminators, multi-label classification, and mask prediction all consume BCE.
