# Notes: Autograd Graph Mechanics Ledger

## Provenance

PyTorch docs *Autograd Mechanism* and the `torch.autograd` FAQ, verified against torch 2.14.1 on CPU:

- **Leaves**: tensors with `requires_grad=False`, plus tensors the user explicitly marks via `requires_grad_(True)`. Any result of a differentiable op is not a leaf. The leaf feeding the graph's input side corresponds to the `AccumulateGrad` node — the place `.grad` is written.
- **`detach`**: returns a new tensor **sharing the same storage**, which is a `requires_grad=False` leaf. `clone` allocates fresh storage — attaching caller data to your graph must go through `clone` (`detach().requires_grad_(True)` silently registers caller memory into your computation).
- **`no_grad` vs `inference_mode`**: outputs of ops under `no_grad` have `requires_grad=False` but are ordinary tensors — you can call `requires_grad_(True)` afterwards and use them as fresh leaves. Outputs under `inference_mode` are **inference tensors**: calling `requires_grad_(True)` on them outside the context raises `RuntimeError: Setting requires_grad=True on inference tensor outside InferenceMode`. The essential difference: no_grad means "don't record this graph", inference_mode means "this tensor can never enter a graph".
- **Grad accumulation**: `backward` does `+=` into leaf `.grad`, never overwrite. Non-scalar outputs need an explicit `gradient=` argument. With the default `retain_graph=False` the graph buffers are freed after the pass; a second `backward` raises `RuntimeError: Trying to backward through the graph a second time`.
- `.grad` itself is a leaf tensor with `requires_grad=False`.

## Key points and traps

1. **Input fidelity**: `x.clone().requires_grad_(True)` and `x.detach().requires_grad_(True)` produce identical ledger values, but the latter shares storage with the input — the `w_copies_input` field (`w.data_ptr() != x.data_ptr()`) is the probe that separates the two roads, and the tests use it to reject the detach road.
2. **Chain order**: the output-side chain of `(w*2+1).sum()` is `SumBackward0 → AddBackward0 → MulBackward0 → AccumulateGrad` — four entries, because `+1` happens after the multiply. First-draft anchors that read "mul, add, sum" drop `AddBackward0`. `next_functions` is a list of `(node, input_nr)` tuples; the leaf side is `AccumulateGrad`.
3. **The gradient ledger is input-independent**: `dy/dw = 2` is constant, so two `backward(ones)` passes give `[2,2] → [4,4]` (accumulation evidence: the second is not `[2,2]`), and the second graph gives `d.grad = [3,3]` regardless of the input values — asserted by `value_independent_grads` over random inputs.
4. **Asymmetric backward arguments**: the first pass must pass `retain_graph=True`, the second must not — removing the first retain raises immediately (8/9 cases fail), which is the observable face of the `retain_graph=False` default.
5. **Two directions of inference_mode**: an inference tensor cannot gain grad afterwards (`t_rg_error=True`); conversely a `no_grad` product can. `t_is_inference()` is a strictly stronger predicate than `requires_grad`.

## Mutation testing (7 effective caught + 1 equivalent)

`clone→detach` (caught by the `w_copies_input` probe), coefficient 2→3, drop `retain_graph`, drop `no_grad`, `inference_mode→no_grad`, swallowing the `t_rg_error` try/except into a hardcoded flag, accumulation turned into overwrite, `d_shared_storage` forced False. **Equivalent mutation**: hardcoding `t_rg_error=True` — the real behavior is True for every legal input, so the hardcoded version is observationally equivalent (which is exactly why the starter forbids hand-filled constants: it skips actually executing step 8).

## Complexity

2–5 element tensors, two backward passes, microseconds; the cost is all in engine semantics.
