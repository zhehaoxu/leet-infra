# Autograd Basics

## Problem

PyTorch's automatic differentiation (autograd) records the dynamic graph formed by the forward computation and, on `backward()`, propagates gradients back along the graph. Mastering the "forward → `backward` → read `.grad`" workflow is a prerequisite for writing any training loop.

## Rules

Implement `solve(x: torch.Tensor, expr: str) -> torch.Tensor`, which **uses autograd** (build the forward graph + call `.backward()`) to return the gradient of the scalar objective with respect to `x`, $\dfrac{\mathrm{d}\sum y}{\mathrm{d}x}$ (the element-wise gradient, with the same shape as `x`):

| expr | forward objective $y$ (element-wise) | analytic gradient |
|---|---|---|
| `"quadratic"` | $3x^2 + 2x$ | $6x + 2$ |
| `"cubic"` | $x^3$ | $3x^2$ |
| `"exp"` | $e^x$ | $e^x$ |

Key requirements:

1. **Do not substitute the analytic formula by hand** — you must go through `backward()` (this problem does not enforce the implementation, but practice honestly).
2. **Do not pollute the input**: the `x` passed in may not have gradient tracking enabled (`requires_grad=False`). Calling `requires_grad_(True)` directly on the argument dirties the caller's tensor; the correct approach is `x.detach().requires_grad_(True)` to create a new leaf.
3. If `expr` is invalid, raise `ValueError`.

## Examples

```
Input: x = [1.0, 2.0], expr = "quadratic"
Output: [8.0, 14.0]        # 6*1+2, 6*2+2
```

## Constraints

- `x` is a float32 tensor of arbitrary shape
- The returned gradient tensor itself must not carry a computation graph (`leaf.grad` satisfies this naturally)
