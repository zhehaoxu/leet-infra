# Solution Notes · Router z-loss

## Why the square of logsumexp

$\mathrm{logsumexp}_e(x_e)$ is the log of the softmax denominator. If a constant $c$ is added to all $x_e$ simultaneously, the softmax is unchanged but the logsumexp increases by $c$. z-loss penalizing "the absolute magnitude of the logsumexp" = penalizing "the +c direction", which is equivalent to pulling the logits toward 0; it does not change the relative shape of the softmax.

## Difference from entropy regularization

Entropy maximization: encourages each row's softmax to be flatter; it destroys the "the model has already decided" signal.
z-loss: does not change the softmax shape, only constrains the absolute value range of the logits; it is lossless with respect to model preferences and purely numerical stabilization.

## logsumexp stability

`torch.logsumexp` internally uses `max + log(sum(exp(x - max)))`, which naturally prevents overflow. Grading does not worry about `exp(1e10)`.
