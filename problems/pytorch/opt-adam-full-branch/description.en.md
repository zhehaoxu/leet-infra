# Adam Full-Branch Update Ledger

Everyone can recite the `torch.optim.Adam` formula; the traps are in the details. The bias
correction counter `t` only counts steps that **actually updated** (a parameter without a
gradient creates no state, so `step` does not advance), `eps` is added **after**
`sqrt(v)/sqrt(1-β₂ᵗ)` (it is not scaled by `sqrt(1-β₂ᵗ)`, unlike the later paper fix-up),
`amsgrad` updates its running maximum **before** it enters the denominator and starts that
state at zeros, `maximize` negates the gradient **before** `weight_decay`, and Adam's
`weight_decay` is **L2-style** (λp joins the gradient and then pollutes both moments, getting
divided by `√v` — the opposite of AdamW's decoupled decay).

This problem mirrors torch 2.14.1's `torch/optim/adam.py`: the five constructor checks in
`Adam.__init__` and the scalar path of `_single_tensor_adam`, over several steps, returning a
step-by-step auditable ledger.

Implement `solve(params, grads, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
amsgrad=False, maximize=False, state=None)`.

## Inputs

- `params`: a real-valued tensor of any shape (float32 / float64).
- `grads`: a sequence of length `T`; each item is a tensor shaped like `params`, or `None`
  (no gradient for that step).
- `lr` / `eps` / `weight_decay`: scalars; `betas`: a two-float tuple.
- `amsgrad` / `maximize`: booleans.
- `state`: `None` (fresh start) or a dict with all four keys: `exp_avg`, `exp_avg_sq`,
  `max_exp_avg_sq` (read only when `amsgrad=True`) and `step` (**number of updates already
  done**, an int).

## Validation

Five gates, raising `ValueError` in this **order**:

```
Invalid learning rate: {}
Invalid epsilon value: {}
Invalid beta parameter at index 0: {}
Invalid beta parameter at index 1: {}
Invalid weight_decay value: {}
```

Each test is `not 0.0 <= x` (plus `< 1.0` for `betas`). Two consequences: **NaN is rejected
too** (`0.0 <= nan` is false, and the message prints `nan`), and `beta == 1.0` is illegal. A
`betas` sequence that is not exactly two entries raises `ValueError`.

## One step

For step `t`:

1. `grads[t] is None`: **skip the whole step** — no state is created, the parameters do not
   move, and the `step` counter **does not advance**.
2. `step += 1` (the counter increments **before** use, so the first update uses `t = 1`).
3. `maximize` is true: `g ← -grads[t]` (**negation first**).
4. `weight_decay != 0`: `g ← g + weight_decay · params` (the parameter value **before** this
   step's update).
5. `m ← β₁·m + (1-β₁)·g`; `v ← β₂·v + (1-β₂)·g²`.
6. `bc1 = 1 - β₁^step`, `bc2 = 1 - β₂^step`, `step_size = lr / bc1`.
7. `denom = sqrt(v) / sqrt(bc2) + eps`; with `amsgrad` first do `mx ← max(mx, v)` and then
   use `sqrt(mx)/sqrt(bc2) + eps` as the denominator.
8. `params ← params + (m / denom) · (-step_size)`.

## Returns

One dict with seven keys:

- `params`: the updated parameters.
- `exp_avg` / `exp_avg_sq`: the final two moments.
- `max_exp_avg_sq`: the running maximum when `amsgrad=True`, otherwise all zeros.
- `path`: `T + 1` parameter snapshots; `path[0]` is a copy of the input.
- `steps`: length `T`, the `step` counter actually used per step (`None` for skipped
  steps) — the bias correction is derived from it.
- `deltas`: length `T`, the actual displacement `params_after - params_before` per step
  (`None` for skipped steps).

Inputs are never modified in place; the returned fields are independent, so mutating
`deltas` in place cannot touch `params`.

## Constraints

- `T` may be 0: `path` then has a single entry, `steps` / `deltas` are empty and both
  moments are zeros.
- Step-by-step agreement with `torch.optim.Adam` at `1e-12` in float64 (with `eps=1e-8` the
  first displacement is `0.099999998` rather than `0.1` — that is how an unscaled eps shows
  up).
- Real tensors only: upstream `is_complex` `view_as_real` plus the `capturable` /
  `differentiable` / `foreach` / `fused` branches are out of scope.

## Examples

```
Input: params=[1.0, -2.0], grads=[[0.5, -1.0]], lr=0.1
Output: params=[0.900000002, -1.900000001], steps=[1], deltas=[[-0.099999998, 0.099999999]]
```

For a single step `m = 0.1g`, `v = 0.001g²`, `bc1 = 0.1`, `bc2 = 0.001`, hence
`denom = |g| + 1e-8` and `step_size = 1`, so the displacement is `-g/(|g|+1e-8)`.

```
Input: params=[1.0, -2.0], grads=[null, [0.5, -1.0]], lr=0.1
Output: steps=[null, 1], path=[[1.0, -2.0], [1.0, -2.0], [0.900000002, -1.900000001]]
```

The first step has no gradient, so no state is created and `step` does not advance; the
second step is the first real update, and its bias correction still uses `t = 1`.
