# Optimizer: Cautious Sign-Consistency Mask (Cautious Optimizer)

## Problem

The update direction \(u_t\) of a momentum optimizer is a running average of past gradients, so on some coordinates \(u_t\) may carry the **opposite** sign of the **current** gradient \(g_t\) — momentum is pushing the parameter "against the latest gradient". Liang et al. (arXiv:2411.16085) found that **a single line of code** — zeroing the coordinates of the update direction whose sign disagrees with the current gradient, then rescaling by the surviving ratio — significantly improves the convergence of Adam/AdamW and friends. This is the **Cautious Optimizer** (e.g. C-AdamW).

This problem asks you to **subclass `torch.optim.Optimizer`** and implement a Cautious AdamW: first compute the AdamW update direction, then apply the cautious mask. `step()` updates each parameter in place; the judge feeds a fixed sequence of gradients and compares the final parameters.

## Rules

- Factory function `solve(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)`, returning an instance of a `torch.optim.Optimizer` subclass whose `step()` updates `params` in place.
- At step \(t\) (counted from 1) with gradient \(g\), first compute the **update direction** \(u\) exactly as AdamW does (not yet multiplied by \(lr\)):
  1. Decoupled weight decay (wd≠0, applied first): \(p \leftarrow p(1-lr\cdot\text{wd})\)
  2. \(m \leftarrow \beta_1 m+(1-\beta_1)g\); \(v \leftarrow \beta_2 v+(1-\beta_2)g^2\)
  3. \(bc_1=1-\beta_1^{\,t},\ bc_2=1-\beta_2^{\,t}\), direction \(u=\dfrac{m/bc_1}{\sqrt{v}/\sqrt{bc_2}+\epsilon}\)
  4. **Cautious mask**: \(M=\mathbb{1}[\,u\odot g>0\,]\) (elementwise indicator that \(u\) and \(g\) share a sign), \(M \leftarrow M\cdot\dfrac{N}{\sum M+1}\) (\(N\) is the number of elements of this parameter)
  5. Rescale and update: \(u \leftarrow u\odot M\); \(p \leftarrow p - lr\cdot u\)
- The rescaling denominator of \(M\) is \(\sum M+1\) (to guard against division by zero); implement it exactly this way. Parameters with `grad is None` are skipped entirely. Everything is float32; calling a ready-made optimizer is forbidden.

## Example

A single parameter \(p=[1.0,\,-2.0,\,0.5]\), `lr=0.1`, `betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0.01`, fed three steps of gradients:

```
g₁ = [ 0.10, -0.20,  0.30]
g₂ = [-0.40,  0.05,  0.20]
g₃ = [ 0.15, -0.10, -0.25]
p after three steps ≈ [0.978047, -1.828335, 0.326713]
```

## Constraints

- float32 dtype, in-place updates; mixing 1D and multi-dimensional parameters is supported
- Must agree step by step with the reference implementation (see notes.md for the comparison setup)
