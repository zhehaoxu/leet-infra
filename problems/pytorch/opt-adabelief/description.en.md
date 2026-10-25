# Optimizer: AdaBelief Belief-Based Second Moment (Belief in Gradient Prediction)

## Problem

AdaBelief (arXiv:1904.03502) keeps Adam's first moment \(m\), but replaces the second-order quantity from "the square of the gradient" with "**the square of the error of the gradient relative to its momentum prediction**" \((g-m)^2\). Intuitively, if the gradient is well predicted by the momentum (small error), then \(s\) is small, the denominator is small, and the effective learning rate is amplified; conversely, when the noise is large the error is large, the denominator grows, and the step tightens automatically. The paper claims this combines fast convergence with flat minima.

This problem asks you to **subclass `torch.optim.Optimizer`** and implement AdaBelief (the non-rectified variant), updating parameters in place via `step()`; the judge feeds a fixed gradient sequence and compares the final parameters against an independent reference element by element.

## Rules

- Implement the factory function `solve(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16, weight_decay=0.0)`, returning an instance of a `torch.optim.Optimizer` subclass; `params` is a list of tensors to optimize and `step()` must **update those very tensors in place**.
- For each parameter \(p\) with a gradient \(g\) at step \(t\) (1-based), with states \(m, s\) initialized to zero:
  1. **Decoupled weight decay** (only when `weight_decay`≠0, before computing the direction): \(p \leftarrow p\,(1 - lr\cdot\text{wd})\)
  2. First moment: \(m \leftarrow \beta_1 m + (1-\beta_1)\, g\)
  3. **Belief second moment** (using the updated \(m\)): \(s \leftarrow \beta_2 s + (1-\beta_2)(g-m)^2 + \epsilon\)
  4. Bias-correction factors: \(\hat\beta_1^{(t)} = 1-\beta_1^t\), \(\sqrt{\hat\beta_2^{(t)}}=\sqrt{1-\beta_2^t}\)
  5. Denominator: \(\text{denom} = \sqrt{s}/\sqrt{\hat\beta_2^{(t)}} + \epsilon\)
  6. Update: \(p \leftarrow p - \dfrac{lr}{\hat\beta_1^{(t)}}\cdot \dfrac{m}{\text{denom}}\)
- \(\epsilon\) is added both into the accumulation of \(s\) and into the final denominator (consistent with library implementations). Everything is float32. Parameters with `grad is None` are skipped entirely (not even wd is applied). **Forbidden** to call an existing optimizer; implement it by hand.

## Examples

Initialize a single parameter \(p=[1.0,\,-2.0,\,0.5]\), with `lr=0.1`, `betas=(0.9, 0.999)`, `eps=1e-16`, `weight_decay=0.01`, and feed three consecutive gradients:

```
g₁ = [ 0.10, -0.20,  0.30]
g₂ = [-0.40,  0.05,  0.20]
g₃ = [ 0.15, -0.10, -0.25]
after three steps p ≈ [0.966739, -1.765123, 0.247988]
```

## Constraints

- Input tensors have dtype float32; the output is updated in place with unchanged dtype
- Supports parameters of any shape; each parameter keeps its own \((m, s)\) state
- Fully consistent with the independent reference step by step (conventions in notes.md)
