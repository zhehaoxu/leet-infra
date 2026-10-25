# Optimizer: Lion Sign-Based Momentum Update

## Problem

Lion (arXiv:2302.06675, Google) was discovered automatically by symbolic search. Each of its update steps is nothing but **the sign of some quantity along the coordinate axes**, so every coordinate moves by exactly \(lr\) in magnitude — there is no per-coordinate adaptive denominator as in Adam, and the state is a single momentum buffer, roughly half of Adam's memory. The core is to interpolate the "current momentum" with the "current gradient" once and take the sign of that interpolation as the update direction.

This problem asks you to **subclass `torch.optim.Optimizer`** and implement Lion, updating parameters in place through `step()`; the judge feeds a fixed sequence of gradients and checks whether the final parameters match an independent reference implementation element by element.

## Rules

- Implement the factory function `solve(params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0)`, returning an instance of a `torch.optim.Optimizer` subclass; `params` is the list of tensors to optimize and `step()` must update those tensors themselves **in place**.
- For each parameter \(p\) with a gradient at step \(t\), gradient \(g\) and momentum buffer \(m\) (initialized to zero):
  1. **Decoupled weight decay** (only when `weight_decay`≠0, before computing the direction): \(p \leftarrow p\,(1 - lr\cdot\text{wd})\)
  2. Interpolated direction: \(u = \beta_1 m + (1-\beta_1)\, g\)
  3. Take the sign: \(u \leftarrow \operatorname{sign}(u)\)
  4. Update the momentum (using the **old** \(m\) and the current \(g\)): \(m \leftarrow \beta_2 m + (1-\beta_2)\, g\)
  5. Parameter update: \(p \leftarrow p - lr\cdot u\)
- `sign(0)=0`. All computation is float32, with one \(m\) per parameter. Parameters with `grad is None` are skipped entirely. Calling any ready-made optimizer is **forbidden**; write it by hand.

## Example

Initialize a single parameter \(p=[1.0,\,-2.0,\,0.5]\), `lr=0.1`, `betas=(0.9, 0.99)`, `weight_decay=0.01`, and feed three consecutive gradient steps:

```
g₁ = [ 0.10, -0.20,  0.30]
g₂ = [-0.40,  0.05,  0.20]
g₃ = [ 0.15, -0.10, -0.25]
p after three steps ≈ [0.897103, -1.894106, 0.398801]
```

## Constraints

- Input tensors are float32; the output is updated in place and the dtype is unchanged
- Supports parameters of any shape; each coordinate moves by exactly \(lr\) per step (signed update) or 0
- Must match the independent reference step by step exactly (see notes.md for the setup)
