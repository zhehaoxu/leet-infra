# Cross-Entropy Loss in Depth

## Definition

For a single sample, the true distribution is a one-hot vector and the prediction is the softmax probability $p$:

$$ H(y,p)=-\sum_{c} y_c\log p_c = -\log p_{\text{true}} $$

$$ \operatorname{CE}=-\log\frac{e^{z_{\text{true}}}}{\sum_j e^{z_j}} = -z_{\text{true}}+\log\sum_j e^{z_j} $$

`reduction="mean"` then averages over the $N$ samples.

## Why merge softmax and log (numerical stability)

Naive `log(softmax(z))`: when $z$ is large, $e^{z}$ overflows to inf, softmax normalizes to 0/1, and the log then yields $-\infty$/NaN. The stable trick is to subtract the row maximum $m$ first:

$$ \log\text{softmax}(z)=\underbrace{(z-m)}_{\text{shifted}}-\log\sum_j e^{z_j-m} $$

Subtracting $m$ does not change the softmax (numerator and denominator are both multiplied by $e^{-m}$), but it guarantees $e^{z_j-m}\le1$ does not overflow, and at least one term equals $e^0=1$, so the denominator is $\ge1$ and $\log0$ is avoided. This is exactly what this problem's reference solution does.

## Three implementation steps

1. Stable log-softmax (subtract the max + `logsumexp`).
2. `gather(-1, target)` to extract the log-prob of the true class column.
3. Negate and reduce according to `reduction`.

Pitfall: `target` must be `long` and shaped `(N,1)` for gather; the class dimension is last.

## Relation to label smoothing / ignore_index

- **label smoothing**: replaces one-hot with $(1-\epsilon)$ + uniform $\epsilon/C$, turning CE into a convex combination; not required here but a common extension.
- **ignore_index** (e.g. -100): `F.cross_entropy` skips those samples, and mean averages only over valid samples. Not covered in this problem; `target` contains only valid classes.

## Gradient easter egg

The gradient of softmax+CE is surprisingly clean: $\partial\mathcal L/\partial z = p - y$ (predicted probability minus one-hot). This is the deep reason it became the standard for classification.

## Related

The log-softmax problem provides the stable kernel, the softmax problem is its normalized form, one-hot is the dense representation of labels, and `nll_loss` is this problem in the form that already takes log-probs.
