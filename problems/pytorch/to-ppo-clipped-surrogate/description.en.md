# PPO / GRPO: Clipped Surrogate Objective

## Problem

In policy optimization, directly doing gradient ascent on the advantage $A$ scaled by the probability ratio of the new policy to the old policy, $\rho=\frac{\pi_\theta}{\pi_{\theta_\text{old}}}$, breaks stability as soon as a single sample is updated too far and $\rho$ drifts away from 1. **PPO**'s clipped surrogate objective adds a trust region $[1-\epsilon,\,1+\epsilon]$ around $\rho$:

$$\mathcal{L}^{\text{CLIP}} = \mathbb{E}\Big[\min\big(\rho_t A_t,\;\ \text{clip}(\rho_t,\,1-\epsilon,\,1+\epsilon)\,A_t\big)\Big]$$

The effect of the $\min$: when the advantage is positive (the action is good), once $\rho$ climbs past $1+\epsilon$ it is no longer rewarded (the bonus is capped, preventing aggressive updates); when the advantage is negative, once $\rho$ falls below $1-\epsilon$ it is no longer pushed down further — i.e. "the step size is limited only in the improving direction". GRPO reuses the same objective; only the advantage $A$ comes from "normalizing rewards within a group" instead (this problem supplies $A$ directly). **Note that $\rho=\exp(\log\pi_\theta - \log\pi_{\theta_\text{old}})$**: the implementation is always given log-probabilities. When optimized as a loss, take the negative.

## Rules

Given a batch ($N$): `logp_new`, `logp_old` (both $(N,)$, log-probabilities), `advantage` ($(N,)$), and the scalar `epsilon` $\epsilon$:

1. Probability ratio $\rho_i = \exp($`logp_new[i]` $-$ `logp_old[i]`$)$
2. Clipped ratio $\hat\rho_i = \mathrm{clip}(\rho_i,\,1-\epsilon,\,1+\epsilon)$
3. Per-sample objective $= \min(\rho_i A_i,\ \hat\rho_i A_i)$
4. loss $= -($ mean of ③ $)$

Return a 0-dim `torch.Tensor` = the batch-mean PPO-clip loss (already negated, ready for minimization).

## Example

```
epsilon = 0.2
logp_new = [-1.0, -1.5, -0.7, -2.0]
logp_old = [-1.2, -1.3, -0.9, -1.8]
advantage = [1.0, -1.0, 0.5, -2.0]

ρ = exp(diff) = exp([0.2, -0.2, 0.2, -0.2]) = [1.2214, 0.8187, 1.2214, 0.8187]
clip(ρ,0.8,1.2) = [1.2, 0.8187, 1.2, 0.8187]
  i0 A>0: min(1.2214*1, 1.2*1)      = 1.2
  i1 A<0: min(0.8187*-1, 0.8187*-1) = -0.8187
  i2 A>0: min(1.2214*0.5, 1.2*0.5)  = 0.6
  i3 A<0: min(0.8187*-2, 0.8187*-2) = -1.6375
mean = -0.16405 → loss = 0.164048
```

## Constraints

- CPU float32; $\epsilon \in (0,1)$.
- $\rho$ is obtained by taking `exp` of the log-probability difference; the probability ratio is never given directly.
- Clipping only acts in the direction of the **upper bound when $A>0$ and the lower bound when $A<0$** (handled automatically by $\min$); return the negated scalar.
