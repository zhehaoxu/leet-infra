# Where lr_scheduler.step() Sits in the Training Loop

## Problem

Whether `optimizer.step()` or `scheduler.step()` comes first is a matter of the `last_epoch` counter — it decides whether the first entry of the schedule gets "skipped" and thereby changes the actual training trajectory; `ReduceLROnPlateau` is even more special since it is driven by a metric stream. This problem runs the same loss under three loop placements and reconciles three ledgers: the per-step effective lr, the final weight, and the scheduler's internal state.

## Rules

Implement `solve()`, run the three variants from the starter docstring, and return the ledger dict:

1. **Variant A** (recommended order): each step first records the current `param_groups[0]["lr"]`, then `zero_grad → backward → opt.step() → sch.step()`.
2. **Variant B** (scheduler jumping the gun): `sch.step()` runs before the lr is recorded; the official UserWarning should be captured (`warned_skipping_first`).
3. **Variant C**: `ReduceLROnPlateau(mode='min', factor=0.5, patience=1, min_lr=0.3, threshold=1e-4)` fed with the metric sequence \([1.0,1.1,\dots,1.6]\); record the lr after each `schC.step(m)`.

`StepLR(step_size=1, gamma=0.5)`; \(loss=w^2\), \(w_0=1\), SGD(lr=0.1). The effective lr sequences are hand-computable: in A, step \(i\) uses \(lr=0.1\cdot0.5^i\); in B everything shifts one slot, \(lr=0.1\cdot0.5^{i+1}\); C is decided by the patience/min_lr state machine.

## Examples

```
Input: none
Output:
{
 "lrs_opt_first": [0.1, 0.05, 0.025, 0.0125],
 "lrs_sched_first": [0.05, 0.025, 0.0125, 0.00625],
 "w_opt_first": 0.6669000387191772, "w_sched_first": 0.8232046365737915,
 "last_epoch_A": 4, "last_epoch_B": 4, "warned_skipping_first": true,
 "plateau_lrs": [1.0, 1.0, 0.5, 0.5, 0.3, 0.3, 0.3],
 "plateau_best": 1.0, "plateau_num_bad": 0, "plateau_last_epoch": 7
}
```

Deriving C: the metrics monotonically worsen → `num_bad` increments each step, and lr drops at the step where the counter passes `patience=1` (the 3rd and 5th metrics); the second drop from \(0.5\) to \(0.25\) hits the `min_lr=0.3` floor, and the third attempt (\(0.15\)) is likewise pinned at \(0.3\).

## Constraints

- lr/metric ledgers compared exactly (\(0.1\cdot0.5^i\) is bitwise identical to the framework's internal computation in IEEE); final weights with `atol=1e-5, rtol=1e-5`.
- Both StepLR variants end with the same `last_epoch` but different lr ledgers — do not use `last_epoch` as the effective lr sequence.
