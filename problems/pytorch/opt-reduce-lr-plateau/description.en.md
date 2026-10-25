# Scheduler: ReduceLROnPlateau Metric-Driven State Machine

## Problem

Most of the `LRScheduler` family has a closed form \(lr(t)\): once the total step count is known, every rate can be precomputed. The **reduce on plateau** branch does the opposite - it never looks ahead and only watches a **validation metric stream**: after the metric fails to improve significantly often enough, the learning rate is multiplied by `factor`. So it is not a function but a **state machine** whose whole state is the triple \((best, num\_bad, cooldown\_counter)\), and the learning rate is a staircase rather than a curve. PyTorch has shipped this scheduler since 0.2, and it is the only member of the family whose `step()` takes an argument.

Implement this state machine with semantics **bit-identical** to the native `torch.optim.lr_scheduler.ReduceLROnPlateau`. The judge feeds no gradients: it pushes a fixed stream of metric values into `step(metrics)` and reads `param_groups[0]["lr"]` after every step, comparing the staircase element by element against the native reference and an independent recomputation.

## Rules

- Factory function `solve(params, base_lr=0.1, factor=0.5, patience=2, min_lr=0.0, mode="min", threshold=1e-4, threshold_mode="rel", cooldown=0, eps=1e-8)`: build the optimizer internally with `torch.optim.SGD(params, lr=base_lr)` and return an `LRScheduler` subclass instance over it; `step(metrics)` only changes learning rates and never touches the parameter tensors.
- Initial state: \(best=+\infty\) (`mode="min"`) or \(-\infty\) (`mode="max"`), \(num\_bad=0\), \(cooldown\_counter=0\). Construction leaves the learning rate at `base_lr`.
- "Significant improvement" test \(\text{is\_better}(cur, best)\) - the `threshold` filters out cosmetic progress:
  - `min` + `rel`: \(cur < best\cdot(1-threshold)\); `min` + `abs`: \(cur < best - threshold\)
  - `max` + `rel`: \(cur > best\cdot(1+threshold)\); `max` + `abs`: \(cur > best + threshold\)
- Every `step(metrics)` executes strictly in this order (swapping any two lines changes the result):
  1. \(cur \leftarrow\) `float(metrics)` (0-dim tensors must work too)
  2. If \(\text{is\_better}(cur,best)\): \(best\leftarrow cur,\ num\_bad\leftarrow 0\); otherwise \(num\_bad \leftarrow num\_bad+1\)
  3. If \(cooldown\_counter>0\): \(cooldown\_counter\leftarrow cooldown\_counter-1,\ num\_bad\leftarrow 0\) (bad steps inside the cooldown are discarded)
  4. If \(num\_bad > patience\) (**strictly** greater; equality does not reduce): perform one reduction, then \(cooldown\_counter\leftarrow cooldown,\ num\_bad\leftarrow 0\)
- A reduction per parameter group: \(new = \max(old\cdot factor,\ min\_lr)\), and \(new\) is written back **only if** \(old-new > eps\); otherwise the reduction is ignored and the old rate stays (the bookkeeping of step 4 still happens).
- Calling `torch.optim.lr_scheduler.ReduceLROnPlateau` or any other ready-made scheduler is **forbidden**.

## Examples

With `base_lr=0.1`, `factor=0.5`, `patience=2`, `mode="min"`, `threshold_mode="rel"`, `cooldown=0`, feeding the metric stream
`[1.0, 0.9, 0.95, 0.9, 0.895, 0.7, 0.75, 0.76, 0.77, 0.77, 0.2, 0.3, 0.3, 0.3, 0.3]`, the lr after each step is:

```
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025]
 # 0.9 equals the previous best so it is not an improvement; 0.895 beats best=0.9 relatively -> counter reset
 # only at step 9 does num_bad=3>2 trigger the first reduction; at step 11 the record 0.2 clears the counter again
```

## Constraints

- Metrics are floats or 0-dim tensors; all arithmetic in plain Python floats
- \(0<factor<1\); `patience` and `cooldown` are non-negative integers
- The learning-rate staircase must be non-increasing, always \(\ge min\_lr\), and match the native reference step by step (see notes.md for the cross-check convention)
