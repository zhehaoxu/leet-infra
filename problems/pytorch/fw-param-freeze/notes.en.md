# Explained: Parameter Freezing, Head Swap and Optimizer Groups

## Origin

PyTorch docs `torch.optim.Optimizer`: an optimizer only knows the parameter tensors passed to its constructor; `torch.nn.Module.requires_grad_(False)` only changes autograd behavior and does **not** remove the parameter from `model.parameters()`. The optimization FAQ and the transfer-learning tutorial prescribe exactly this recipe: "freeze the backbone + replace the head + `[p for p in model.parameters() if p.requires_grad]`". SGD's in-buffer update is `buf = momentum*buf + grad; param -= lr*buf` (dampening=0).

## Key points and pitfalls

1. **Frozen ≠ removed**: `total_before=124` (the total is unchanged by freezing), `trainable_before=0`. If you hand all of `model.parameters()` to SGD, the frozen ones are skipped by `step` since `grad=None`, but they are **already in param_groups**, so `load_state_dict` on the optimizer will accept extra empty state groups — the correct habit is filtering by `requires_grad`.
2. **Head swap changes the parameter ledger**: \(6\times4+4=28 \to 6\times5+5=35\), total \(124\to131\). Flags recorded via `named_parameters()` order show that the second Linear inside `nn.Sequential` is keyed `encoder.2` (module index), not `encoder.1`.
3. **Frozen parameters have `.grad is None`**: a `requires_grad=False` leaf is not part of the backward graph at all, so `zero_grad` never fills it — this is why `frozen_grad_is_none=True`, and it is also the mechanism behind "frozen layers consume no gradient memory".
4. **`len(opt.state)`**: optimizer state is lazily created on the first `step()` — `SGD(momentum=0.9)` creates one buffer per passed parameter (2 trainable parameters → 2 state entries); plain momentum-free SGD stores nothing (empirically `len(opt.state)==0`). The count thus verifies two things at once: only trainable parameters were passed, and momentum is actually on.
5. **The criterion for "bitwise unchanged"**: use `torch.equal`, not allclose — a frozen parameter has no reason to differ by even one ULP.

## Cross-check basis

The solution runs the full autograd + optimizer loop; the judge's `_oracle` never touches `nn`/autograd: pure-Python double loops recompute the forward (`math.tanh`), the analytic gradient \(\partial L/\partial W_{ij}=\frac{2}{N}\sum_b (f(x_b)-y_b)_i h_{bj}\), and a hand-written momentum recurrence; after 3 steps it compares against `final_head_*` (atol/rtol=1e-5). Counts, flags and booleans are compared exactly against hardcoded values. Additional assertions: parameter-count delta of 7 after the head swap, and the first-step momentum overshoot rebound.

## Complexity

The model is tiny (131 parameters, 2 samples, 3 steps); the whole flow is microsecond-level; the oracle's double loops are \(O(B\cdot(48+36+30)\cdot 3)\) and instantaneous.
