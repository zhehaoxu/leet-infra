# Custom autograd.Function: Forward/Backward Ledger

## Problem

`torch.autograd.Function` is the front door for plugging your own CUDA/fused kernel into the backward graph. The new API moves the responsibility of saving intermediates from `forward` to `setup_context`, and `ctx.save_for_backward` decides which tensors get released after backprop; `forward` may also return non-differentiable objects that never enter the graph. This problem asks you to implement a fused operator \(y=c\,x^2\tanh(x)\) (with an `int` second output) and keep a ledger of the forward/backward call behavior.

## Rules

Implement `solve()` and follow the fixed flow in the starter docstring (input \(x=[-2.5,-0.75,0.5,1.25,3.0]\), \(c=2.0\), upstream gradient \(g=[1,2,3,4,5]\)):

1. New-style API: `forward(x, c)` takes **no ctx**; all context is built in `setup_context(ctx, inputs, output)`; \(x\) goes through `ctx.save_for_backward`, \(c\) is attached directly as `ctx.c`.
2. The second output `neg_count` is a plain Python `int`: the framework passes it through verbatim (`neg_count_type == "int"`), it builds no graph, and the corresponding gradient argument in `backward` is `None` (`extra_grad_is_none == True`).
3. In `backward`, `ctx.needs_input_grad` must be `[True, False]` (\(c\) is non-differentiable); return `(gx, None)`.
4. The hand-written gradient must agree with "the same expression as a composition of standard ops + autograd" (`atol=rtol=1e-5`).
5. Backpropagating the same graph twice: the second attempt must raise `RuntimeError` (saved tensors already released), yet the `backward` entry counter still increments — `bwd_calls` records **how many times the entry was called**.

## Examples

```
Input: none
Output:
{
 "y_flat": [-12.33267879486084, -0.71454256772995, 0.23105858266353607,
            2.650886297225952, 17.910985946655273],
 "neg_count": 2, "neg_count_type": "int", "y_has_grad_fn": true,
 "grad_x_flat": [10.198546409606934, 5.15321159362793, 3.9523744583129883,
                 20.47085952758789, 60.59123229980469],
 "grad_x_ref_flat": [10.198546409606934, 5.15321159362793, 3.9523744583129883,
                     20.47085952758789, 60.591224670410156],
 "grads_match_ref": true, "extra_grad_is_none": true,
 "needs_input_grad": [true, false], "fwd_calls": 2, "bwd_calls": 3,
 "second_backward_raises": true
}
```

The last elements of `grad_x_flat` and `grad_x_ref_flat` differ by one ULP (\(60.59123\) vs \(60.59122\)): the hand-written analytic form and the chain-rule graph evaluate in different floating-point orders — expected.

## Constraints

- Numeric comparison uses `atol=1e-5, rtol=1e-5`; booleans/counts/type-names must match exactly.
- Do not mix the old-style `forward(ctx, ...)` with `setup_context` — under the new API that is an immediate `TypeError`.
- No randomness in the whole flow.
