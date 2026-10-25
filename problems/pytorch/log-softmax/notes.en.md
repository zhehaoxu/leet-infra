# Numerically Stable Log-Softmax Explained

## Why subtract the maximum

Softmax contains $e^{x}$: in float32 it becomes `inf` at $x\gtrsim 88$ and `0` at $x\lesssim -87$.
The naive `log(softmax(x))` therefore produces a batch of `inf/inf=nan` or `log(0)=-inf`. Use translation invariance:
$$ \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i-m}}{\sum_j e^{x_j-m}} $$
With $m=\max_j x_j$ all exponent arguments are $\le 0$, the largest term is $e^0=1$, so nothing overflows and the denominator is $\ge 1$, so nothing underflows either.

## Writing the log form directly is more stable

Cross entropy is `log_softmax` followed by `nll`. If you compute `softmax` first and then `log`, small probabilities underflow to 0 before the `log`, giving $-\infty$; `F.cross_entropy`/`log_softmax` fuse the two steps and emit in the log domain directly,
which is optimal for both numerical accuracy and speed — the reason frameworks recommend `CrossEntropyLoss`
instead of manually stitching `Softmax + NLL`.

## A gradient coincidence

The gradient of softmax cross entropy with respect to log-softmax is extremely clean: $\partial/\partial x_i = p_i - y_i$
(predicted probability minus one-hot label). This "subtract the one-hot" step is the implementation core of the
backward pass for classification losses, and it also explains why fusing log and softmax into one operator saves memory and makes differentiation easier.

## Echoes with other problems

- Shares the subtract-maximum trick with the "softmax" problem; log_softmax merely rewrites the final division as a subtraction in the log domain.
- Related to "attention-softmax-temp": temperature scaling followed by log_softmax gives temperature-conditioned log probabilities.
- FlashAttention's online softmax (also in this collection) is essentially a stable log-sum-exp that "maintains a running max/sum while blocking",
  the streaming version of this problem's idea.
