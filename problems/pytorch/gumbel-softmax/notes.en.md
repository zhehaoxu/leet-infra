# Gumbel-Softmax Explained

## Gumbel-max: turning categorical sampling into taking a max

If $G_i\sim\text{Gumbel}(\log p_i,1)$ are independent, then the distribution of $\arg\max_i G_i$ is exactly
$p$. This is the "Gumbel-max trick": adding noise and taking the max is equivalent to sampling according to $p$. And
$$ G_i = \log p_i + \tilde g_i,\quad \tilde g_i=-\log(-\log u_i),\ u_i\sim U(0,1) $$
so starting from `logits` (i.e. the unnormalized form of $\log p$), adding standard Gumbel noise and then taking the argmax is sampling.

## From "hard" to "soft": switching to softmax so gradients can flow back

argmax is not differentiable. Gumbel-Softmax (Jang et al. 2016; Maddison et al. 2016) relaxes the argmax into a
temperature softmax: $y=\text{softmax}((\log\text{it}+g)/\tau)$. As $\tau\to0$, $y$ approaches the sampled
one-hot; for finite $\tau$ it is a differentiable soft approximation through which gradients can flow — exactly what lets discrete VAEs / NAS architecture encodings /
hard attention train end to end.

## Reparameterization = moving the randomness outside

The core idea: concentrate the "randomness" in external noise $u$ (independent of the parameters), making $y$ a deterministic function of the parameters.
Passing $u$ as an explicit input in this problem is precisely this idea, and it also makes grading reproducible. A similar trick is the VAE's
$z=\mu+\sigma\epsilon$ (moving the noise onto $\epsilon$).

## The role of temperature and annealing pitfalls

- The smaller $\tau$, the closer to a true discrete sample, but the higher the gradient variance and the easier it saturates; the larger $\tau$, the smoother but the "mushier".
- Training often **anneals** $\tau$ (from large to small) to balance exploration against approaching hard samples.
- The Straight-through (ST) variant: use a hard one-hot in the forward pass and soft gradients in the backward pass, combining unbiasedness with differentiability.
