# Explained: Forward/Backward Hooks — Capturing Activations and Rewriting Gradients

## Origin

PyTorch docs `torch.nn.Module.register_forward_hook` / `register_full_backward_hook` / `register_backward_hook`: the forward hook signature is `(module, input, output)`, and **a non-None return value replaces the module output** (for a single output, return the tensor directly; returning a `(t,)` tuple would turn the output into a tuple). The `register_full_backward_hook` signature is `(module, grad_input, grad_output)`; the returned tuple **replaces grad_input** and keeps propagating upstream; the docs state it only affects the gradient path flowing to inputs and requires at least one input/output with `requires_grad` to fire. The legacy `register_backward_hook` emits a FutureWarning and may miss some grad_inputs when the forward contains multiple autograd nodes — this problem lets it only count without rewriting, precisely to safely test "both hook styles attach and count".

## Key points and pitfalls

1. **An output replacement counts in both the forward and backward ledgers**: after the tanh hook doubles its output, forward is \(out=W_2\cdot(2\tanh(a_1))\); backward passes through that mul-by-2 node, so \(dL/dt=2\,dL/dh\) — you cannot change only the forward.
2. **A backward hook only rewrites grad_input**: fc1's `2*g+1` rewrite hits `x.grad`, but `fc1.weight.grad = outer(dL/da1, x)` and `fc1.bias.grad = dL/da1` travel along different edges and stay clean — the grader nails this down via "parameter gradients still equal the analytic values".
3. **The length of `gin`**: `grad_input` in the hook corresponds to the input gradients received by the module's Function; for `F.linear` it contains only the 1 item for the data input (parameter gradients do not pass through here); the returned tuple length must match `gin` — changing the length raises `RuntimeError: hook 'hook' has changed the size of value`.
4. **What the forward hook sees in `inp`** is the actual argument passed into the module's forward: fc2's `inp[0]` is the replaced \(2\tanh\), which is the "replacement took effect" probe; `acts["tanh"]` records the **original** out the hook received.
5. **Firing-count ledger**: with one forward+backward pass all five counters equal 1. An extra step or a missing hook registration breaks it.

## Cross-check basis

The solution uses real hooks + autograd; the judge's `_oracle` is pure Python: double loops with `math.tanh` hand-derive forward \(\to\) loss \(\to\) explicit chain rule (including the ×2 of the replacement node and the 2g+1 of the hook), compared per tensor with `atol=rtol=1e-5`; fired counts compared exactly; plus three invariants: "fc2_input = 2*tanh", "bias.grad not polluted by 2g+1", "x.grad is not the raw value".

## Complexity

A single forward/backward through a tiny 3→2→1 network — microsecond level; the difficulty is entirely the directionality of hook semantics.
