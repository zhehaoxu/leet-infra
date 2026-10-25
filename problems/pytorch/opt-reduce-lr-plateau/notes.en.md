# In Depth: ReduceLROnPlateau Metric-Driven State Machine

## Source
- The authoritative reference is the PyTorch implementation itself:
  `torch/optim/lr_scheduler.py::ReduceLROnPlateau` in `.venv-eval` (`best=mode_worse` in `__init__`, the four-stage
  order inside `step`, `_reduce_lr`'s `max(old*factor, min_lr)` plus the `old-new>eps` test, the `in_cooldown`
  property, and the four `mode` x `threshold_mode` branches of `_is_better`). This scheduler has no paper;
  these exact semantics have shipped since PyTorch 0.2 (2017), so the source was read line by line.
- Conceptually it is the classical hill-climbing "back off when there is no progress" heuristic
  (the reduce-on-plateau notation in the Sutton & Barto textbook), but do **not** implement the intuitive version:
  it differs from the source in three places - the patience boundary, when the cooldown counter is decremented,
  and what `eps` actually suppresses.
- Item #90(3) of todos.md gives no arXiv id (it only says "cross-checks directly against native"), and this
  problem follows exactly that plan.

## Core points and pitfalls
1. **`num_bad > patience` is strict**. With `patience=2` two bad steps are tolerated and the third one reduces;
   writing `>=` shifts every reduction one step early.
2. **The first metric only sets the baseline**. `best` starts at ±inf, so step 1 is always judged "better" and
   `num_bad` stays 0 - it can never reduce; with `patience=0` the earliest possible reduction is step 2.
3. **`threshold` means "improvement must clear a dynamic bar"**: `min/rel` compares against `best*(1-threshold)`,
   so a zero improvement (metric unchanged) counts as worse and increments `num_bad`. `abs` mode uses
   `best-threshold`. In `mode="max"` all four comparisons flip direction.
4. **Bookkeeping timing inside the cooldown**: while `cooldown_counter>0` you first do `-=1` and then `num_bad=0`,
   discarding this step's bad count; and `cooldown_counter` is set to `cooldown` only **after** a reduction, so the
   reducing step itself is not cooled and the following `cooldown` steps are swallowed. Forgetting `num_bad=0`
   causes a burst of over-reductions.
5. **`eps` suppresses the write-back, not the bookkeeping**: with `new=max(old*factor, min_lr)`, if
   `old-new <= eps` the lr is not written at all, yet `num_bad`/`cooldown_counter` are still reset. This is exactly
   what makes every reduction after hitting the `min_lr` floor an idempotent no-op.
6. **The `min_lr` floor is a max applied after the multiplication**, not an adaptive `factor`; once lr lands on the
   floor it stays there for all later reductions.
7. `step()` only mutates `param_groups[*]["lr"]` and never the parameter tensors; the judge reads both the raw
   tensors and the lr sequence to confirm no over-reach.

## Cross-check convention
- External reference: the **native `torch.optim.lr_scheduler.ReduceLROnPlateau`**. Same `SGD(lr=base_lr)` start,
  same metric stream, compare `param_groups[0]["lr"]` after every step with tolerance 1e-15 (five configurations
  covering all four `mode` x `threshold_mode` quadrants, `patience=0/1/2`, `cooldown=0/2/3`, a `min_lr` floor and
  `eps=1e-3`, each over a fixed 40-metric stream).
- Independent recomputation: `_oracle` walks the same semantics through a differently written pure-Python state
  machine (thresholds as reference ratio / reference difference, floor via a conditional instead of `max()`),
  tolerance 1e-12, catching solutions that copied the native structure but reordered it.
- The example numbers were produced by actually running the reference solution, which is bit-identical to the
  native reference.
- Degenerate cases / invariants: the hand-traced `cooldown=2` staircase; idempotence once on the `min_lr` floor;
  a first reduction fully swallowed by `old-new<=eps`; the `patience=0` baseline rule; 0-dim tensor metrics;
  lr non-increasing and `\ge min_lr`; and the reduction ledger (exactly 5 reductions).

## Complexity
Each `step(metrics)` is \(O(1)\) (a constant number of float comparisons per parameter group), with no state
tensors and no randomness; a stream of \(T\) metrics costs \(O(T)\), all in Python floats, so the whole suite
decides in milliseconds.
