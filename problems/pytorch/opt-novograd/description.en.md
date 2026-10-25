# Optimizer: NovoGrad Layer-wise Gradient Normalization

## Problem

Adam uses the element-wise second moment \(v\) as the denominator, so its memory overhead is on the same order as the parameters, and it scales each coordinate independently. Ginsburg et al. (arXiv:1905.11286) proposed **NovoGrad**: reduce the second moment to a single scalar, the "sum of squared gradients over the whole layer", use it to normalize the whole layer's gradient, then stack ordinary momentum on top, and treat weight decay with the same "normalize first, then decouple" approach. Its memory needs only one momentum plus one scalar per layer, which makes it especially practical for large batches and large models; it is one of NVIDIA's default optimizers when training ResNet.

This problem asks you to **subclass PyTorch's `torch.optim.Optimizer` base class** and implement a NovoGrad that updates parameters one by one through `step()`. The grader constructs several tensors to optimize, repeatedly feeds a fixed gradient sequence and calls `step()`, then compares the final parameters element-wise against the reference implementation.

## Rules

- Implement the factory function `solve(params, lr=1e-3, betas=(0.95, 0.98), eps=1e-8, weight_decay=0.0, grad_averaging=False)`, returning an instance of a `torch.optim.Optimizer` subclass; `params` is the list of tensors to optimize. The instance's `step()` must **update the tensors in `params` in place**.
- Let the \(t\)-th step (\(t\) counted from 1), the gradient of a parameter \(p\) be \(g\), and the **sum of squared gradients over the whole layer** for that parameter be \(\lVert g\rVert^2=\sum_i g_i^2\) (a scalar). Each parameter maintains a scalar \(s\) (the layer second-moment EMA) and an \(m\) of the same shape (momentum):
  1. Layer second moment: on the first step \(s \leftarrow \lVert g\rVert^2\); afterwards \(s \leftarrow \beta_2 s + (1-\beta_2)\lVert g\rVert^2\)
  2. Normalized gradient: \(\hat g = \dfrac{g}{\sqrt{s}+\epsilon}\)
  3. Decoupled weight decay (only when `weight_decay`≠0): \(p \leftarrow p\,(1-lr\cdot\text{wd})\)
  4. Momentum: on the first step \(m \leftarrow \hat g\); afterwards (if `grad_averaging`, first \(\hat g\leftarrow(1-\beta_1)\hat g\)) \(m \leftarrow \beta_1 m + \hat g\)
  5. Parameter update: \(p \leftarrow p - lr\cdot m\)
- **Note that NovoGrad has no bias correction**: neither the momentum nor the denominator is divided by \(1-\beta^t\); the step size is just `lr`, and the coefficient in front of \(\hat g\) is \(1\) (only `grad_averaging=True` multiplies by \((1-\beta_1)\)).
- Parameters with `grad is None` must be skipped as-is (not even weight decay is applied). float32 throughout. Using a ready-made `torch.optim` optimizer is forbidden.

## Examples

A single parameter \(p=[1.0,\,-2.0,\,0.5]\), `lr=0.01`, `betas=(0.95, 0.98)`, `eps=1e-8`, `weight_decay=0.01`, `grad_averaging=False`, fed three steps of gradients:

```
g₁ = [ 0.10, -0.20,  0.30]
g₂ = [-0.40,  0.05,  0.20]
g₃ = [ 0.15, -0.10, -0.25]
after three steps p ≈ [1.008826, -1.984079, 0.473279]
```

## Constraints

- Input dtype is float32; in-place update, dtype unchanged
- Supports a mix of 1D and multi-dimensional parameters (for multi-dimensional parameters, \(s\) is a scalar obtained by **summing over all elements of the whole layer**)
- Must match the reference implementation exactly step by step (grading details in notes.md)
