# Optimizer: Adafactor Factored Second Moment

## Problem

Adam must maintain, for every parameter, a second moment \(v\) of the same shape, which doubles memory. Adafactor (Shazeer & Stern, arXiv:1804.04235) approximates the \(v\) of a matrix parameter by the **outer product of two low-rank row and column statistics**, shrinking the second-moment state from \(O(rc)\) to \(O(r+c)\); it also replaces the fixed learning rate with **RMS update clipping** and optionally scales the step by the parameter RMS. It is one of the default optimizers for models such as T5.

This problem asks you to **subclass `torch.optim.Optimizer`** and implement Adafactor (using the common conventions of its BigVision variant), updating parameters in place via `step()`; the judge feeds a fixed gradient sequence and compares the final parameters with an independent reference implementation element by element. To keep judging stable on CPU, all computation is done in **float32** (no bfloat16 momentum, and `relative_step` is disabled).

## Rules

Implement the factory function `solve(params, lr=1e-2, betas=(0.9, 0.999), eps1=1e-30, eps2=1e-3, decay_rate=-0.8, clip_threshold=1.0, scale_parameter=True, weight_decay=0.0)`, returning an instance of a `torch.optim.Optimizer` subclass; `step()` must update `params` in place.

Let the group step counter \(t\) start at 1, with \(\beta_1,\beta_2^{cap}\) taken from `betas`. For each parameter \(p\) that has a gradient \(g\):

1. Per-step second-moment decay: \(\beta_{2,t} = \min\!\big(\beta_2^{cap},\; 1 - t^{\text{decay\_rate}}\big)\), and set \(b = 1 - \beta_{2,t}\).
2. Effective learning rate: if `scale_parameter`, \(lr_{eff} = \max(\epsilon_2, \operatorname{RMS}(p))\cdot lr\), otherwise \(lr_{eff}=lr\); here \(\operatorname{RMS}(x)=\sqrt{\frac{1}{n}\sum_i x_i^2}\).
3. Squared gradient: \(U = g^2 + \epsilon_1\).
4. Second moment:
   - **Matrix parameters (\(g\) has \(\ge 2\) dimensions; take the last two as rows \(r\) and columns \(c\))**: the row and column statistics each get their own EMA
     \[ v_r \leftarrow (1-b)v_r + b\cdot \operatorname{rowmean}(U),\quad v_c \leftarrow (1-b)v_c + b\cdot \operatorname{colmean}(U) \]
     reconstruct \(\hat v_{ij} = \dfrac{v_{r,i}}{\sum_k v_{r,k}}\cdot v_{c,j}\).
   - **Non-scalar, dimensionality \(<2\)**: one block-wide EMA \(v \leftarrow (1-b)v + b\,U\), with \(\hat v = v\).
5. Preconditioned direction: \(u = \big(\hat v^{-1/2}\big)\odot g\).
6. **RMS update clipping**: \(r_u=\operatorname{RMS}(u)\), \(\text{factor}=\max(r_u/\text{clip\_threshold},\,1)\), \(u \leftarrow u/\text{factor}\cdot lr_{eff}\).
7. First moment: \(m \leftarrow \beta_1 m + (1-\beta_1) u\), then take \(u \leftarrow m\).
8. Decoupled weight decay (when `weight_decay`≠0): \(p \leftarrow p\,(1 - lr_{eff}\cdot\text{wd})\).
9. Update: \(p \leftarrow p - u\).

States start at zero; parameters with `grad is None` are skipped. Calling an existing Adafactor/Adam is **forbidden**; write it by hand.

## Examples

A matrix parameter \(p=\begin{bmatrix}0.5 & -0.2\\ 0.3 & 0.4\\ -0.1 & 0.2\end{bmatrix}\), with `lr=0.1`, `betas=(0.9,0.999)`, `scale_parameter=True`, `weight_decay=0`, and three gradient steps:

```
g₁ = [[ 0.2,-0.1],[ 0.1, 0.3],[-0.2, 0.05]]
g₂ = [[-0.1, 0.25],[ 0.2,-0.15],[ 0.05, 0.1]]
g₃ = [[ 0.15,-0.05],[-0.2, 0.1],[ 0.1,-0.3]]
after three steps p ≈ [[0.489838, -0.202574], [0.292380, 0.392157], [-0.093338, 0.196794]]
```

## Constraints

- Input is float32; updated in place, dtype unchanged
- Rows and columns are both accumulated over the last two dimensions; the factored path targets **matrix parameters (dimensionality exactly 2)**, and \(\sum_k v_{r,k}\) sums over the entire row-statistics vector (on the first step \(b=1\) and the squared gradient is positive, so the denominator is nonzero); dimensionality 1/0 uses the block-wide EMA
- Must match the independent reference exactly, step by step (conventions and cross-check policy in notes.md)
