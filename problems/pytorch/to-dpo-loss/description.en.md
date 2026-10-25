# DPO: Direct Preference Optimization Loss

## Problem

**DPO** (Rafailov et al. 2023) skips the whole RLHF pipeline of "train a reward model first, then do RL" and optimizes the policy model directly on preference data as a classification objective. The core insight: the optimal policy of RLHF satisfies $\pi^*(y|x)\propto\pi_\text{ref}(y|x)\exp(r(y,x)/\beta)$, so conversely the reward can be expressed **implicitly** through the log-likelihood ratio of the policy and the reference policy; the reward model thus disappears, leaving a Bradley–Terry style logistic loss over preference pairs $(y_w \succ y_l)$:

$$\mathcal{L}_\text{DPO} = -\log\sigma\!\Big(\beta\big[(\log\tfrac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)}) - (\log\tfrac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)})\big]\Big)$$

Intuition: make "how much chosen improves over the reference model" larger than "how much rejected improves over the reference model". $\beta$ controls the strength of the deviation from the reference model. This problem directly gives four **sequence-level log probabilities** (i.e. the sum of the token logprobs over the whole answer), so you only have to assemble the formula above.

## Rules

Given a batch (length $N$) of: `chosen_logp`, `chosen_ref_logp`, `rejected_logp`, `rejected_ref_logp` (all $(N,)$ float), and a scalar `beta` $\beta$:

1. chosen implicit reward term $=$ `chosen_logp` $-$ `chosen_ref_logp`
2. rejected implicit reward term $=$ `rejected_logp` $-$ `rejected_ref_logp`
3. logits $= \beta\times($ ① $-$ ② $)$, shape $(N,)$
4. per-sample loss $= -\log\sigma(\text{logit})$, using a **stable** log-sigmoid: $\log\sigma(x)=\min(0,x)-\log(1+e^{-|x|})$
5. take the **mean** over the batch

Return a scalar `torch.Tensor` (0-dimensional) = the batch-mean DPO loss.

## Examples

```
beta = 0.1
chosen_logp       = [-1.2, -0.8, -2.0]
chosen_ref_logp   = [-1.0, -1.0, -1.5]
rejected_logp     = [-2.5, -1.9, -1.0]
rejected_ref_logp = [-1.4, -1.6, -1.2]

① = [ -0.2, 0.2, -0.5 ]
② = [ -1.1, -0.3,  0.2 ]
logit = 0.1 * (① - ②) = 0.1*[0.9, 0.5, -0.7] = [0.09, 0.05, -0.07]
loss  = -mean(logsigmoid([0.09,0.05,-0.07])) = 0.682126
```

## Constraints

- CPU float32; the inputs are sequence-level logprobs (already summed over tokens, no further accumulation needed).
- The log-sigmoid must be numerically stable (equivalent to `F.logsigmoid`); do not write `log(sigmoid(x))` directly and cause underflow.
- Return a 0-dimensional scalar tensor.
