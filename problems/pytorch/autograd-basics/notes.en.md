# Autograd Basics Explained

## What automatic differentiation is

For every tensor participating in an operation with `requires_grad=True`, PyTorch **dynamically builds a computation graph** that records the forward operator chain; `.backward()` propagates back along the graph and accumulates $\partial(\text{scalar})/\partial(\text{leaf})$ into `.grad`. There is no need to hand-write derivative formulas.

## The three steps in this problem

1. **Create the leaf**: `leaf = x.detach().requires_grad_(True)`.
2. **Forward**: compose $y$ according to `expr` (quadratic $3x^2+2x$, cubic $x^3$, exp $e^x$).
3. **Backward**: `y.sum().backward()`, then read `leaf.grad`.

## Two essential details

- **Why `.detach()`**: the input `x` may itself carry a `grad_fn` or already be referenced by another graph. `detach()` cuts the link to the old graph, and `requires_grad_(True)` turns it into a **new leaf** — this both satisfies "differentiate with respect to x" and **does not pollute the tensor passed in by the caller** (it does not write its `.grad` or change its `requires_grad`). The statement explicitly forbids polluting `x`; calling `x.requires_grad_(True)` directly would modify an external tensor.
- **Why `.sum()`**: `.backward()` by default computes the gradient of a **scalar**. When $y$ has the same shape as $x$, `y.sum().backward()` yields `leaf.grad[i] = ∂y_i/∂x_i` (each component differentiated with respect to its own input), which is exactly the element-wise derivative the problem asks for. Calling `y.backward()` directly without summing fails because the output is non-scalar and requires a `gradient=` argument.

## Verification against the analytic derivatives

Analytic derivatives: $\frac{d}{x}(3x^2+2x)=6x+2$; $(x^3)'=3x^2$; $(e^x)'=e^x$. The autograd result should match.

## Pitfalls

- `backward()` runs only once: a second backward pass over the same graph requires `retain_graph=True`.
- `.grad` is **accumulated**; repeated backward calls without zeroing add up (this is why training loops call `zero_grad`).
- To exclude a section from differentiation, use `torch.no_grad()` / `detach()`.

## Related

optimizer-training-loop is the minimal "build graph → backward → step" loop; the forward passes of problems such as lstm-cell / attention can all have gradients computed automatically by autograd.
