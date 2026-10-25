# Optimizer: ADOPT Bias-Correction-Free + Previous-Step Second Moment + Gradient Clipping

## Problem

Taniguchi et al. (arXiv:2411.02853) proved that Adam converges only when \(\beta_2\) is sufficiently close to 1, and that it needs bias correction. They proposed **ADOPT**: **drop the bias correction**, normalize the current gradient with the second moment from the **previous step**, and **clip the normalized gradient with a bound that loosens with the step count** — provably converging at the optimal rate for any \(\beta_2\). In large-model training it is known for being "more stable, unafraid of the value of \(\beta_2\)".

This problem asks you to **subclass `torch.optim.Optimizer`** and implement ADOPT, with `step()` updating each parameter in place. The judge feeds a fixed gradient sequence and compares the final parameters.

## Rules

- Factory function `solve(params, lr=1e-3, betas=(0.9, 0.9999), eps=1e-6, weight_decay=0.0)`, returning an instance of a `torch.optim.Optimizer` subclass; `step()` updates `params` in place.
- Each parameter keeps \(m=0,\ v=0\); at step \(t\) (1-based) with gradient \(g\):
  1. Decoupled weight decay (wd≠0, done first): \(p \leftarrow p\,(1-lr\cdot\text{wd})\)
  2. **First step (\(t=1\))**: only set \(v \leftarrow g^2\) and **end this step immediately** (no update to \(m\), no update to \(p\)).
  3. Thereafter (\(t\ge 2\)), first use the \(v\) **accumulated on the previous step**:
     - Denominator \(d = \max(\sqrt{v},\ \epsilon)\) (element-wise `clamp(min=eps)`)
     - Normalize and clip: \(\tilde g = \operatorname{clip}\!\left(\dfrac{g}{d},\ \pm\, t^{1/4}\right)\)
     - Momentum: \(m \leftarrow \beta_1 m + (1-\beta_1)\tilde g\)
     - Update: \(p \leftarrow p - lr\cdot m\)  (**no bias correction**, \(m\) is used directly)
     - Only then update the second moment: \(v \leftarrow \beta_2 v + (1-\beta_2) g^2\)
- Key ordering: **first use the previous step's \(v\) to normalize and update the parameters, and only at the very end fold the current \(g^2\) into \(v\)**; the first step does not update the parameters. Parameters with `grad is None` are skipped entirely. Everything is float32 and calling an existing optimizer is forbidden.

## Examples

A single parameter \(p=[1.0,\,-2.0,\,0.5]\) with `lr=0.01`, `betas=(0.9,0.9999)`, `eps=1e-6`, `weight_decay=0.01`, fed five gradient steps (the first step does not update the parameter):

```
g₁ = [ 0.10, -0.20,  0.30]
g₂ = [-0.40,  0.05,  0.20]
g₃ = [ 0.15, -0.10, -0.25]
g₄ = [ 0.05,  0.30, -0.10]
g₅ = [-0.20, -0.10,  0.40]
after five steps p ≈ [1.000569, -2.000692, 0.499016]
```

## Constraints

- dtype float32, updated in place; supports mixed 1D and multi-dimensional parameters
- Must match the reference implementation exactly, step by step (cross-check convention in notes.md)
