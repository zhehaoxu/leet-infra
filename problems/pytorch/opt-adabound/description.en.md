# Optimizer: AdaBound Dynamic Learning-Rate Clamping (Adaptive Bounds → SGD)

## Problem

Adaptive optimizers (Adam and friends) converge fast, but on many tasks they generalize worse than SGD. AdaBound (arXiv:1902.11010, Microsoft) offers a "best of both worlds" recipe: wrap Adam's **per-coordinate effective learning rate** in a pair of bounds \([\alpha_t,\beta_t]\) that tighten as the step count grows, so that they **converge from the two sides onto the same constant \( \text{final\_lr} \)** — early on the bounds are very wide and the behavior is close to Adam; later they collapse to a single point and the behavior degenerates into SGD with a fixed learning rate, which combines fast early convergence with good final generalization.

This problem asks you to **subclass `torch.optim.Optimizer`** and implement AdaBound, updating parameters in place via `step()`; the judge feeds a fixed gradient sequence and compares the final parameters against an independent reference element by element.

## Rules

- Implement the factory function `solve(params, lr=1e-3, final_lr=0.1, betas=(0.9, 0.999), gamma=1e-3, eps=1e-8, weight_decay=0.0)`, returning an instance of a `torch.optim.Optimizer` subclass; `params` is a list of tensors to optimize and `step()` must **update those very tensors in place**. `base_lr` is the initial `lr` of each parameter group taken at construction time.
- For each parameter \(p\) with gradient \(g\), states \(m, v\) (initialized to zero), at step \(t\) (incremented only when that parameter has a gradient):
  1. Bias-correction factors: \(\hat\beta_1^{(t)}=1-\beta_1^{\,t}\), \(\sqrt{\hat\beta_2^{(t)}}=\sqrt{1-\beta_2^{\,t}}\)
  2. Dynamic bounds (clamping **the reciprocal of the effective learning rate**): \( \text{flr}=\text{final\_lr}\cdot lr / \text{base\_lr}\), \(\ \text{lower}=\text{flr}\big(1-\tfrac{1}{\gamma t+1}\big),\ \text{upper}=\text{flr}\big(1+\tfrac{1}{\gamma t}\big)\)
  3. Step-size scalar: \(\text{step}=\dfrac{lr\cdot\sqrt{\hat\beta_2^{(t)}}}{\hat\beta_1^{(t)}}\)
  4. **Decoupled weight decay** (only when `weight_decay`≠0, before the direction is computed): \(p \leftarrow p\,(1 - lr\cdot\text{wd})\)
  5. First/second moments: \(m \leftarrow \beta_1 m + (1-\beta_1)g\); \(v \leftarrow \beta_2 v + (1-\beta_2)g^2\)
  6. Denominator: \(\text{denom}=\sqrt{v+10^{-15}}+\epsilon\)
  7. Per-coordinate clamping, then update: \(\text{update}=\operatorname{clamp}\!\big(\text{denom}/\text{step},\ \text{lower},\ \text{upper}\big)\odot m\), \(p \leftarrow p - \text{update}\)
- Notes: the library first adds the constant \(10^{-15}\) to \(v\) and only then takes the square root (for numerical stability), and \(\epsilon\) is added to the final denominator on top of that. Everything is float32. Parameters with `grad is None` are skipped entirely. Calling an existing optimizer is **forbidden**.

## Examples

Initialize a single parameter \(p=[1.0,\,-2.0,\,0.5]\) with `lr=0.1`, `final_lr=0.1`, `betas=(0.9, 0.999)`, `gamma=0.01`, `eps=1e-8`, `weight_decay=0.01`, and feed five consecutive gradients:

```
g₁=[ 0.10,-0.20, 0.30]  g₂=[-0.40, 0.05, 0.20]  g₃=[ 0.15,-0.10,-0.25]
g₄=[ 0.05, 0.30,-0.10]  g₅=[-0.20,-0.10, 0.40]
after five steps p ≈ [1.048022, -1.981064, 0.400048]
```

## Constraints

- Input tensors have dtype float32; the output is updated in place and the dtype is unchanged
- Supports parameters of any shape; each parameter keeps its own \((m, v)\)
- The bounds clamp \(\text{denom}/\text{step}\) **independently for every coordinate** (not the resulting step size)
- Must match the independent reference exactly, step by step (for the convention see notes.md)
