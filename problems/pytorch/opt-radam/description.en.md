# Optimizer: RAdam Rectified Adaptive Moment Estimation

## Problem

In the early phase of training, Adam's variance estimate is very noisy, so the adaptive learning rate jitters violently and a manual warmup is often needed for stability. RAdam (arXiv:1908.03213) **internalizes** warmup: it explicitly estimates the variance of the second moment and uses a **stability coefficient \(r_t\)** that varies with the step count to "rectify" the learning rate — when the early variance is unreliable, \(r_t\) approaches or even disables the adaptive term, and once the simple moving average length \(n_t\) is large enough it switches back to standard Adam.

This problem asks you to **subclass `torch.optim.Optimizer`** and implement RAdam (the `degenerated_to_sgd=False` variant), updating parameters in place via `step()`; the grader feeds a fixed gradient sequence and compares the final parameters element-wise against an independent reference.

## Rules

- Implement the factory function `solve(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, n_sma_threshold=5)`, returning an instance of a `torch.optim.Optimizer` subclass; `params` is the list of tensors to optimize, and `step()` must **update these tensors in place**.
- For each parameter \(p\) with gradient \(g\), state \(m, v\) (initially zero), at step \(t\) (\(t\) counted from 1; **each parameter counts independently and increments only when that parameter has a gradient**):
  1. First moment: \(m \leftarrow \beta_1 m + (1-\beta_1)\, g\)
  2. Second moment: \(v \leftarrow \beta_2 v + (1-\beta_2)\, g^2\)
  3. Effective average length: let \(n_{\max}=\dfrac{2}{1-\beta_2}-1\), \(\beta_2^t=\beta_2^{\,t}\), \(n_t = n_{\max}-\dfrac{2\,t\,\beta_2^{\,t}}{1-\beta_2^{\,t}}\)
  4. **Rectified branch** (when \(n_t \ge \texttt{n\_sma\_threshold}\)): stability coefficient
     \[ r_t=\sqrt{\frac{(1-\beta_2^{\,t})(n_t-4)}{n_{\max}-4}\cdot\frac{n_t-2}{n_t}\cdot\frac{n_{\max}}{n_{\max}-2}} \]
     step size \(\text{step}=\dfrac{lr\cdot r_t}{1-\beta_1^{\,t}}\), update \(p \leftarrow p - \text{step}\cdot \dfrac{m}{\sqrt{v}+\epsilon}\).
  5. **Below the threshold**: \(r_t=-1\) makes the step size negative, in which case **the whole step is skipped** — \(p\) is left untouched and no weight decay is applied (a direct manifestation of RAdam's built-in warmup).
- Weight decay is decoupled (only when `weight_decay`≠0, **and only on steps where the rectified branch actually takes effect**): \(p \leftarrow p\,(1 - lr\cdot\text{wd})\). float32 throughout. Parameters with `grad is None` are skipped entirely. Calling a ready-made optimizer is **forbidden**.

## Examples

Initialize a single parameter \(p=[1.0,\,-2.0,\,0.5]\), `lr=0.1`, `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.01`, `n_sma_threshold=5`, and feed **8 steps** of gradients consecutively (the first 4 steps barely move because the threshold is not reached):

```
g₁=[0.10,-0.20,0.30]   g₂=[-0.40,0.05,0.20]  g₃=[0.15,-0.10,-0.25] g₄=[0.05,0.30,-0.10]
g₅=[-0.20,-0.10,0.40]  g₆=[0.30,0.10,-0.20]  g₇=[-0.10,0.25,0.05]  g₈=[0.20,-0.15,-0.30]
after eight steps p ≈ [0.992954, -1.998415, 0.503093]
```

## Constraints

- Input tensor dtype is float32; updated in place, dtype unchanged
- Supports parameters of any shape; each parameter keeps its own \((m, v)\) and an independent step counter
- Steps below the threshold must be a complete no-op (not even weight decay)
- Exactly consistent with the independent reference step by step (details in notes.md)
