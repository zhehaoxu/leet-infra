# Optimizer: AdamW Decoupled Weight Decay

## Problem

Adam adds \(L_2\) regularization straight into the gradient, so the regularization term is also scaled by the adaptive denominator and the real strength of the weight decay drifts with the gradient scale. Loshchilov & Hutter (arXiv:1711.05101) proposed **AdamW**: **decouple** the weight decay from the gradient and shrink the parameter itself directly and proportionally, so that \(L_2\) regularization and the adaptive learning rate each do their own job. It is the default optimizer of modern large-model training.

This problem asks you to **subclass PyTorch's `torch.optim.Optimizer` base class** and implement an AdamW that updates parameters one by one through its `step()` — rather than writing a pure function. The judge constructs several tensors to optimize, hands them to your optimizer, repeatedly feeds a fixed gradient sequence and calls `step()`, then compares the final parameters against the reference implementation (native `torch.optim.AdamW`, `amsgrad=False`) element by element.

## Rules

- Implement the factory function `solve(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)`, returning an instance of a `torch.optim.Optimizer` subclass; `params` are the tensors to optimize (a list of plain `torch.Tensor`). The instance's `step()` must **update those very tensors in `params` in place** (the judge reads the original tensors to check the values).
- Replicate `torch.optim.AdamW` exactly (`amsgrad=False`, no `maximize`). For each parameter \(p\) with gradient \(g\) at step \(t\) (1-based):
  1. **Decoupled weight decay** (only when `weight_decay`≠0, and it happens before the moment estimates): \(p \leftarrow p\,(1 - lr\cdot\text{wd})\)
  2. First moment: \(m \leftarrow \beta_1 m + (1-\beta_1) g\)
  3. Second moment: \(v \leftarrow \beta_2 v + (1-\beta_2)\, g^2\)
  4. Bias correction: \(\hat m = \dfrac{m}{1-\beta_1^{\,t}},\quad \hat v = \dfrac{v}{1-\beta_2^{\,t}}\)
  5. Parameter update: \(p \leftarrow p - \dfrac{lr}{1-\beta_1^{\,t}}\cdot\dfrac{m}{\sqrt{v}/\sqrt{1-\beta_2^{\,t}} + \epsilon}\)
- Parameters with `grad is None` must be skipped as-is (not even the weight decay is applied).
- All computation is float32; each parameter's `step` counter is maintained independently. Instantiating or calling `torch.optim.AdamW` / `torch.optim.Adam` directly is **forbidden**; write the update by hand.

## Examples

Initialize a single parameter \(p=[1.0,\,-2.0,\,0.5]\) with `lr=0.1`, `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.01`, and feed three consecutive gradients:

```
g₁ = [ 0.10, -0.20,  0.30]
g₂ = [-0.40,  0.05,  0.20]
g₃ = [ 0.15, -0.10, -0.25]
after three steps p ≈ [0.971888, -1.786759, 0.276560]
```

## Constraints

- Input tensors have dtype float32; the output is updated in place and the dtype is unchanged
- Supports mixed parameter lists with 1D (biases / normalization gains) and multi-dimensional (weight matrices) tensors
- Must be exactly consistent with native `torch.optim.AdamW`, step by step (cross-check convention in notes.md)
