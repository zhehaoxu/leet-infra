# Gumbel-Softmax Sampling (Reparameterization)

## Problem

To sample from a categorical distribution $\text{softmax}(\text{logits})$ **differentiably**, plain `argmax`/`multinomial`
is a non-differentiable discrete operation. Gumbel-Softmax uses reparameterization to move the randomness onto external noise: add
standard Gumbel noise to the logits and pass them through a temperature softmax to obtain a "soft one-hot" sample through which gradients can flow. It is used for VAE discrete latent variables,
neural architecture search (NAS), and hard attention.

> To keep grading **deterministic**, this problem takes the uniform random numbers `u` as an explicit input instead of sampling inside the function.

## Rules

- Implement `solve(logits, u, tau=1.0)`; `logits` has shape `(..., K)`, and `u` has the same shape and holds uniform random
  numbers in $(0,1)$.
- Gumbel transform: $g = -\log(-\log(u))$ (standard Gumbel$(0,1)$ noise).
- Output: $y = \text{softmax}((\text{logits} + g)/\tau)$ along the last dimension.
- As $\tau\to 0$ it approaches one-hot (equivalent to Gumbel-max sampling); larger $\tau$ is smoother.

## Examples

```
Input: logits = [2, 1, 0.5, 0.1]
     u = [0.5, 0.9, 0.1, 0.3], tau = 1
  g = -log(-log(u)) = [0.36651, 2.25037, -0.83403, -0.18563]
  logits+g = [2.367, 3.250, -0.334, -0.086] → /1 → softmax
Output: [0.2798, 0.6773, 0.0188, 0.0241]   (sums to 1; class 1 is boosted by its large noise)
```

## Constraints

- float32 input; `u` strictly inside $(0,1)$ (do not pass 0 or 1, which would give log(0))
- Each output row sums to 1; for very small `tau` the output approximates one-hot
