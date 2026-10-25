# Explained: Custom autograd.Function Forward/Backward Ledger

## Origin

PyTorch docs *Extending PyTorch → Defining a custom Function*: since 1.8/1.9 the **new API** is recommended — `forward` without `ctx`, plus a new `setup_context(ctx, inputs, output)` that owns `save_for_backward`. The docs note that mixing the old-style `forward(ctx, ...)` with `setup_context` fails signature binding with `TypeError: missing a required argument` — the very source of constraint #4. Non-Tensor outputs (the Python `int` here) are passed through by the engine without building any graph; the gradient argument at that position in `backward` is `None`; `ctx.needs_input_grad` tells whether each input needs a gradient and returning `None` for unneeded inputs is the documented style. Tensors in `save_for_backward` are released after backward by default (`retain_graph=False`), so backpropagating again raises RuntimeError.

## Key points and pitfalls

1. **Pick one of the two APIs**: a class defining `setup_context` must use `forward(x, c)`; writing `forward(ctx, x, c)` throws `TypeError` at `.apply` — that is exactly the behavior the constraint encodes.
2. **The right way to emit a non-differentiable second output**: a Python `int` needs no `mark_non_differentiable` (it is not a Tensor and never enters the graph); a Tensor returned without marking would be treated as differentiable, forcing backward to handle its gradient. `type(neg_count).__name__ == "int"` is the probe for this behavioral difference.
3. **The `bwd_calls == 3` ledger**: apply twice, first graph backpropagated once; for the second graph we grad twice in a row, and the second attempt **still entered `backward`** (entry counter already incremented) — the "graph freed" RuntimeError is only raised when accessing `ctx.saved_tensors`. This explains the counter-intuitive `fwd_calls=2, bwd_calls=3`.
4. **Hand-written backward vs autograd reference**: the analytic form `c*(2x*tanh + x²*(1-tanh²))*g` differs from the chain-rule graph at ULP level (last element 60.59123229980469 vs 60.591224670410156) — precisely why the cross-check uses `allclose`, not `torch.equal`.
5. **Division of labor between `ctx.c = c` and `save_for_backward`**: plain tensors not derived from inputs can be attached directly to ctx, but **inputs or views of inputs** must go through `save_for_backward`, otherwise autograd cannot detect in-place memory reuse/version conflicts.

## Cross-check basis

The judge's `_oracle` is a pure-Python `math` layer: it recomputes \(y\) and the analytic gradient element-wise with `math.tanh` (float64) and compares against the solution's float32 ledger with `atol=rtol=1e-5`; `neg_count`, flags and counters are compared exactly against hardcoded values. Three more invariant assertions: "int output builds no graph", "needs_input_grad ledger", "re-backprop raises".

## Complexity

A 5-element tensor with two applies — microsecond-level; the real cost is understanding engine behavior.
