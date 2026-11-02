# notes (opt-sgd-full-branch)

## Source

- Code: torch 2.14.1 in the local judging environment, `torch/optim/sgd.py` (same as
  upstream main).
  - `SGD.__init__` (28-82): `lr < 0` (45-46), `momentum < 0` (47-48),
    `weight_decay < 0` (49-50), `nesterov and (momentum <= 0 or dampening != 0)` (63-64).
  - `SGD.step` (106-148): `_init_group` yields `(params, grads, momentum_buffer_list)`,
    and the buffers are written back into `state[p]["momentum_buffer"]` afterwards.
  - `_single_tensor_sgd` (322-379): 344 maximize negation, 346-355 weight_decay,
    357-369 momentum/nesterov, 371-379 parameter update.
  - `_multi_tensor_sgd` (382-460) was checked line by line: it also calls `_foreach_neg`
    first, and also "clone when the buffer is None, otherwise `mul_(momentum).add_(g,
    1-dampening)`" — same semantics.
- Cross-check: `check_matches_torch_optim` uses `torch.optim.SGD` itself as the oracle
  (9 configs × 3 trials × 7 steps, two of them forcing `grad=None`), matching step by step
  at 1e-12 in float64.

## Traps in the upstream implementation

1. **The first momentum step ignores dampening**: `buf = grad.detach().clone()` never goes
   through `alpha=1-dampening`. So with `momentum=0.9, dampening=0.5` the first direction
   is the gradient itself; writing `buf=(1-τ)g` is wrong right away.
2. **An all-zero buffer is not the same as no buffer**: upstream tests whether the state
   dict has a `momentum_buffer` key, not whether the buffer is zero. This problem makes that
   explicit as `buffers=None` (no history) versus a passed all-zero tensor (history), which
   differ by a `(1-dampening)` factor on the first step.
3. **maximize negates before weight_decay**: `grad = grads[i] if not maximize else
   -grads[i]`, and only then `grad.add(param, alpha=weight_decay)`. The effective gradient is
   `-g + λp`, **not** `-(g + λp)`. `_multi_tensor_sgd` negates first as well.
4. **nesterov uses the updated buffer, and adds g rather than the buffer**:
   `buf.mul_().add_()` comes first, then `grad = grad.add(buf, alpha=momentum)`; `grad`
   already carries weight_decay at that point.
5. **With momentum == 0 dampening plays no part** and does not raise — the constructor only
   validates the `nesterov + dampening` combination.
6. **dampening itself is never range checked**: `5.0` and `-0.1` are both legal as long as
   `nesterov=False`. With `dampening=1` the buffer only decays by `momentum` and the
   gradient term has coefficient zero.
7. **The nesterov gate is `momentum <= 0`**: `momentum=0` raises too; but `momentum<0` hits
   `Invalid momentum value` earlier, so in practice gate 4 covers `momentum==0` and
   `dampening != 0`.
8. **The validation order is fixed**: `lr` → `momentum` → `weight_decay` → `nesterov`. With
   several invalid values only the first one is reported.
9. **`lr == 0` is legal**: the parameters stay put while the buffer keeps advancing (there is
   no short circuit for a zero learning rate).
10. **Parameters whose `p.grad is None` are filtered out by `_init_group`** and get no state
    at all. This problem maps `grads[t] is None` to "skip the whole step" — so **the first
    real gradient after a skipped step goes through the first-buffer path again**, ignoring
    dampening once more. This is the nastiest point here: `[None, g]` and `[zeros, g]` give
    different results.
11. **weight_decay uses the parameter value from before this step's update** (the `p` inside
    `grad.add(p, ...)` has not moved yet).
12. **Upstream updates in place** (`param.add_(grad, alpha=-lr)`, `buf.mul_().add_()`); this
    problem never mutates its inputs and rebinds instead.

## Normalizations in this problem

- **"Filter by parameter" becomes a `None` inside `grads`**: upstream skips when
  `p.grad is None` for the single parameter at hand; here "no gradient this step" is
  explicit as `None` in the sequence — the same semantics.
- **Hidden state made explicit**: upstream keeps the buffer in `optimizer.state`; here it is
  the `buffers` argument plus a returned value, with `buffers=None` ⇔ no `momentum_buffer`
  key and a passed tensor ⇔ the key exists (even when all zeros).
- **`dirs` is an added observation surface**: upstream never exposes "which direction was
  actually applied". To tell nesterov / non-nesterov and first-buffer / existing-buffer apart
  without relying on float drift, this problem records `step` too and returns a clone of it
  (mutating it in place does not touch `buffers`).
- **Only Python scalar `lr`**: upstream also accepts a 1-element Tensor (including the
  requires_grad differentiable path); this problem takes the post-`_to_scalar` semantics.
- **No per-group hyper-parameters**: upstream `param_groups` may hold several sets; this
  problem has a single group.

## Mutations

26 mutations: **all 22 substantive ones are CAUGHT**, 4 are classified as equivalent and
each has evidence (1500 random configurations compared byte by byte, 0 differences):

- `m-no-param-clone` (`p = params.clone()` → `p = params`): inside the loop `p` is only
  **rebound** by `p = p - lr*step`, never mutated in place, and `path` clones each snapshot,
  so the input is never written.
- `m-mom-gate-positive` (`momentum != 0` → `momentum > 0`): `momentum < 0` already raises in
  `_check`, so a negative momentum never reaches the loop.
- `m-wd-gate-positive` (`weight_decay != 0` → `> 0`): same reasoning, a negative
  `weight_decay` is rejected by validation.
- `m-no-grad-clone` (`g = g_in.clone()` → `g = g_in`): `-g`, `g + wd*p` and
  `(1-dampening)*g` all allocate new tensors and `g` is only read; the returned `dirs` are
  cloned as well.

One survived the first round and needed an extra assertion:

- `m-dirs-no-clone` (`dirs.append(step.clone())` → `dirs.append(step)`): `buf` is
  **reassigned** every step (`buf = momentum*buf + ...`) rather than accumulated in place,
  so `dirs[i]` holds an older object that naturally differs from the final `buffers`; only in
  the "single step + momentum≠0" case is `step` the buffer itself and the storage shared.
  The `check_no_inplace` assertion "mutating `dirs[0]` in place must leave `buffers`
  untouched" is what catches it.

## Overlap check

- Against `sgd-momentum-step` (order 74): that one pins `dampening=0`, `nesterov=False` and a
  single step, always takes `buffers` as an existing state, and covers neither `maximize` nor
  validation. This problem adds the full branch matrix, multi-step carry-over, `None`
  skipping, the first-buffer rule and the four constructor gates.
- Against `adam-step` (75), `opt-adamw` (182) and `opt-adamp` (189): adaptive denominators,
  bias correction and decoupled weight decay are out of scope; this problem is plain SGD.
- Against `opt-clip-accum` (197): that one is gradient accumulation plus clipping and has no
  momentum buffer or nesterov.
- No other problem in the library covers the "SGD branch matrix plus constructor validation"
  axis.
