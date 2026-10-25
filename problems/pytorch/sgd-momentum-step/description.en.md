# SGD + Momentum Step

## Problem Description

SGD is the baseline optimizer of deep learning; adding **momentum** accumulates inertia along the descent direction and damps oscillation; layering on **weight decay** gives the L2-regularized SGD. This problem asks for a hand-written single update step — given the old parameters, current gradients, and old buffers, return the new parameters and new buffers.

## Rules

- Implement `solve(params, grads, buffers, lr, momentum=0.0, weight_decay=0.0)` and return `(new_params, new_buffers)`.
- **Follow exactly the update formulas of `torch.optim.SGD` (dampening=0, nesterov=False)**:

  1. Fold weight decay into the gradient: $g \leftarrow g + \text{wd}\cdot p$
  2. Momentum buffer: $b \leftarrow \mu\, b + g$ (no buffer when momentum=0)
  3. Parameter update: $p \leftarrow p - \text{lr}\cdot b$ (when momentum=0, use $g$ directly in place of $b$)

- Inputs are immutable (clone them); return new tensors.
- `buffers` initialize to all zeros (PyTorch's lazily created buffer on the first step is also zero).

## Example

```
Input: params=[1.0, 1.0, 1.0], grads=[0.5, -0.5, 2.0], buffers=[0, 0, 0]
       lr=0.1, momentum=0.9, weight_decay=0.0
new_params=[0.95, 1.05, 0.80]
new_buffers=[0.5, -0.5, 2.0]
(first step: buf = 0.9*0 + g = g; p' = p - 0.1*buf)
```

## Constraints

- Support float32 / float64; output dtype matches `params`
- Do not use `torch.optim.SGD`; hand-write the update formulas; must match PyTorch step for step exactly (see notes for cross-checking)
