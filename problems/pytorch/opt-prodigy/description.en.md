# Optimizer: Prodigy Adaptive Learning Rate (D-Adaptation)

## Problem

Tuning the learning rate (\(lr\)) is the most time-consuming part of large-model training. The **D-Adaptation** idea of Defazio & Mishchenko (arXiv:2301.07733) is to **estimate online the "distance \(D\) to the optimum"** and use it to decide the effective step size automatically, with almost no manual \(lr\). **Prodigy** is the practical Adam implementation of this idea — it maintains an estimate \(d\) that grows with the step count and multiplies \(d\) into the effective learning rate, thereby "learning the \(lr\) while training".

This problem asks you to **subclass `torch.optim.Optimizer`** and implement Prodigy, with `step()` updating each parameter in place. The grader feeds a fixed gradient sequence and compares the final parameters. This is the most complex problem in the collection: it involves a **global scalar estimate \(d\)** and **two passes** (statistics first, update second).

## Rules

- Factory function `solve(params, lr=1.0, betas=(0.9, 0.999), d0=1e-6, d_coef=1.0, growth_rate=float('inf'), weight_decay=0.0, eps=1e-8)`, returning an instance of a `torch.optim.Optimizer` subclass whose `step()` updates `params` in place.
- Let \(\beta_3=\sqrt{\beta_2}\). Group-level state: \(d\leftarrow d_0\), \(d_{\max}\leftarrow d_0\), \(\mathit{num}\leftarrow 0\) (scalars, kept across steps); per-parameter state: \(s=0\), the initial snapshot \(p_0=p\), \(m=0\), \(v=0\). Each step:
  1. Effective step size \(d_{lr}=d\cdot lr\) (bias correction is not enabled in this problem).
  2. \(\mathit{num}\leftarrow \beta_3\cdot \mathit{num}\) (**decay first**); also set this step's accumulated denominator \(\mathit{den}=0\).
  3. **First pass** (for each parameter \(p\) with gradient \(g\)):
     - \(\mathit{num}\ \mathrel{+}= \dfrac{d}{d_0}\,d_{lr}\,\langle g,\ p_0-p\rangle\)
     - \(m \leftarrow \beta_1 m + d(1-\beta_1) g\)
     - \(v \leftarrow \beta_2 v + d^2(1-\beta_2) g^2\)
     - \(s \leftarrow \beta_3 s + \dfrac{d}{d_0}\,d_{lr}\, g\)
     - \(\mathit{den}\ \mathrel{+}= \sum_i |s_i|\)
  4. If \(\mathit{den}=0\), do not update the parameters this step and stop. Otherwise \(\hat d = d_{coef}\cdot \mathit{num}/\mathit{den}\) (a scalar), and:
     - if \(d==d_0\): \(d\leftarrow\max(d,\hat d)\)
     - \(d_{\max}\leftarrow\max(d_{\max},\hat d)\); \(d\leftarrow\min(d_{\max},\ d\cdot \texttt{growth\_rate})\)
  5. **Second pass** (for each parameter with a gradient): decoupled weight decay (wd≠0) \(p\leftarrow p(1-d_{lr}\cdot\text{wd})\); denominator \(\text{denom}=\sqrt{v}+d\cdot\epsilon\); \(p\leftarrow p-d_{lr}\cdot\dfrac{m}{\text{denom}}\).
- Note: **the second pass uses the old \(d_{lr}\) from the start of the step**, while the \(d\cdot\epsilon\) in the denominator uses the **new \(d\)** just updated. Parameters with `grad is None` are skipped in both passes. float32 throughout, ready-made optimizers forbidden.

## Examples

A single parameter \(p=[1.0,\,-2.0,\,0.5]\), `lr=1.0`, `betas=(0.9,0.999)`, `d0=1e-2`, `d_coef=1.0`, `growth_rate=1.1`, `weight_decay=0.0`, `eps=1e-8`, fed 30 steps of gradients (\(\texttt{randn}\cdot0.5\) under `torch.manual_seed(0)`). Because \(d\) rises step by step and \(d_{lr}\) grows with it, the parameter barely moves early on and only decreases noticeably later:

```
after 30 steps p ≈ [0.852818, -2.713569, 0.255777]
```

## Constraints

- dtype float32, in-place update; supports a mix of 1D and multi-dimensional parameters (the inner product \(\langle g,p_0-p\rangle\) is computed by flattening and then taking the dot product)
- Must match the reference implementation step by step (grading details in notes.md)
