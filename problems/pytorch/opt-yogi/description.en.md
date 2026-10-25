# Optimizer: Yogi Sign-controlled Second Moment

## Problem

Adam's second moment \(v\) uses an exponential average of \(g^2\); when sparse/bursty gradients appear, \(v\) jumps up instantly and the learning rate is crushed. Zaheer et al. (arXiv:1807.01874) proposed **Yogi**: change the update of \(v\) from an "additive average" to a **sign-controlled addition** — increase or decrease only in the direction of the difference between \(v\) and \(g^2\), with a step of \((1-\beta_2)\), so that \(v\) changes more gently and is more robust to gradient noise. It is an Adam variant commonly used in recommender systems and NLP.

This problem asks you to **subclass `torch.optim.Optimizer`** and implement Yogi, updating each parameter in place through `step()`. The grader feeds a fixed gradient sequence and compares the final parameters.

## Rules

- Factory function `solve(params, lr=1e-2, betas=(0.9, 0.999), initial_accumulator=1e-6, eps=1e-3, weight_decay=0.0)`, returning an instance of a `torch.optim.Optimizer` subclass whose `step()` updates `params` in place.
- Each parameter initializes \(m=v=\texttt{initial\_accumulator}\) (**nonzero**, an element-wise constant fill). At step \(t\) (counted from 1), with gradient \(g\):
  1. Decoupled weight decay (wd≠0, done first): \(p \leftarrow p(1-lr\cdot\text{wd})\)
  2. First moment: \(m \leftarrow \beta_1 m + (1-\beta_1) g\)
  3. Second moment (the core of Yogi): \(v \leftarrow v - (1-\beta_2)\cdot \operatorname{sign}(v - g^2)\cdot g^2\)
  4. Bias correction: \(bc_1=1-\beta_1^{t},\ bc_2=1-\beta_2^{t}\), step size \(\text{step}=lr/bc_1\)
  5. Update: \(p \leftarrow p - \text{step}\cdot\dfrac{m}{\sqrt{v}/\sqrt{bc_2}+\epsilon}\)
- The order of the denominator operators must be \(\sqrt{v}/\sqrt{bc_2}+\epsilon\) (the square root of \(bc_2\) goes into the denominator, and eps is added last). Parameters with `grad is None` are skipped entirely. float32 throughout, ready-made optimizers forbidden.

## Examples

A single parameter \(p=[1.0,\,-2.0,\,0.5]\), `lr=1e-2`, `betas=(0.9,0.999)`, `initial_accumulator=1e-6`, `eps=1e-3`, `weight_decay=0.01`, fed five steps of gradients:

```
g₁ = [ 0.10, -0.20,  0.30]
g₂ = [-0.40,  0.05,  0.20]
g₃ = [ 0.15, -0.10, -0.25]
g₄ = [ 0.05,  0.30, -0.10]
g₅ = [-0.20, -0.10,  0.40]
after five steps p ≈ [1.001245, -1.980141, 0.472945]
```

## Constraints

- dtype float32, in-place update; supports a mix of 1D and multi-dimensional parameters
- Must match the reference implementation step by step (grading details in notes.md)
