# Solution Notes · PPO/GRPO Clipped Surrogate Objective

## How min + clip form a trust region

Consider `clip(ratio)` on its own: it squeezes the ratio into $[1-\epsilon,1+\epsilon]$. But PPO takes $\min(\rho A, \hat\rho A)$ rather than plain $\hat\rho A$ — this guarantees that the objective is a **pessimistic bound**:
- $A>0$: only when $\rho$ moves up past $1+\epsilon$ does the clipped term become the smaller one and take effect, suppressing "over-rewarding good actions"; when the policy gets worse, $\rho A$ is smaller and is naturally selected, so the degradation is not hidden.
- $A<0$: clipping takes effect only when $\rho$ drops below $1-\epsilon$, preventing "over-penalizing bad actions"; when it moves up, the unclipped term is used.
As a result the gradient is cut to 0 when the improving direction goes out of bounds and is left untouched in all other directions, which effectively limits how far the policy can change per step.

## Why pass log-probabilities instead of ratios

Numerically $\log\pi$ can be extremely small, and storing the ratio directly tends to underflow; real implementations (trl / OpenAI baselines) all pass `logprob` and compute `exp(logprob - old_logprob)` inside the loss. This problem follows that convention, and the tests cover the branch where $\rho$ clearly leaves the bounds and clipping takes effect.

## GRPO differs only upstream

GRPO (Group Relative Policy Optimization) removes the critic, samples a group of responses for the same prompt, and uses the standardized within-group reward $(r-\mu)/\sigma$ as the token advantage $A$. The objective is exactly the same as PPO-clip, so this problem provides $A$ directly and treats "within-group standardization" as an external step. Real GRPO also commonly adds a KL regularizer against a reference policy (not included here; a candidate for a follow-up problem).
