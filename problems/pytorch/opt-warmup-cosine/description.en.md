# Scheduler: Linear Warmup + Cosine Annealing Learning Rate

## Problem

When training large or vision models, the learning rate (\(lr\)) is almost never constant: the first few steps need **linear warmup** (start smoothly with a small \(lr\), avoiding early gradient shocks to the second-moment estimate), and then **cosine annealing** lowers the \(lr\) from the peak all the way to near the lower bound. This "warmup + cosine" recipe is practically the default for Transformer training.

This problem asks you to **subclass `torch.optim.lr_scheduler.LRScheduler`** and implement a scheduler: at construction it sets the optimizer's \(lr\) to the value for step \(0\), and each call to `step()` advances to the next step and writes back the new \(lr\). The grader repeatedly reads `param_groups[0]["lr"]` and compares it against the reference sequence.

## Rules

- Factory function `solve(params, base_lr=0.1, warmup_steps=3, total_steps=10, min_lr=0.0)`: internally builds an optimizer with `torch.optim.SGD(params, lr=base_lr)` and returns a scheduler instance inheriting from `LRScheduler`; `base_lr` is the peak (upper bound) of the cosine segment and `min_lr` is the annealing lower bound.
- Let the current step be \(t\) (\(t=0,1,2,\dots\); construction completes at step \(0\)). The piecewise definition of \(lr(t)\):
  - **warmup segment** (\(0\le t<\) `warmup_steps`):
    \[lr(t)=base\_lr\cdot\frac{t+1}{warmup\_steps}\]
    i.e. step \(0\) is \(base\_lr/\text{warmup}\), and the last step (\(t=\text{warmup}-1\)) reaches exactly \(base\_lr\).
  - **cosine segment** (\(t\ge\) `warmup_steps`): let \(\text{denom}=\text{total\_steps}-\text{warmup\_steps}\),
    \[\text{progress}=\operatorname{clamp}\!\left(\frac{t-\text{warmup\_steps}}{\text{denom}},\,0,\,1\right),\qquad
    lr(t)=min\_lr+(base\_lr-min\_lr)\cdot\tfrac12\bigl(1+\cos(\pi\cdot\text{progress})\bigr)\]
    If \(\text{denom}\le 0\) (i.e. `total_steps<=warmup_steps`), take \(\text{progress}=1\) and \(lr=\text{min\_lr}\).
- **After construction** the optimizer's \(lr\) must equal \(lr(0)\); each call to `step()` increments \(t\) by \(1\) and writes \(lr(t)\) back to `param_groups[0]["lr"]`. If `warmup_steps=0`, use cosine throughout (then \(lr(0)=base\_lr\)).
- Calling ready-made `LinearLR`/`CosineAnnealingLR`/`LambdaLR` etc. is forbidden; you must compute it yourself with the formula above and write it back.

## Examples

`base_lr=0.1`, `warmup_steps=3`, `total_steps=10`, `min_lr=0.001`. Reading the \(lr\) from step \(0\) to step \(9\):

```
[0.033333, 0.066667, 0.100000, 0.100000, 0.095098, 0.081363, 0.061515, 0.039485, 0.019637, 0.005902]
```

(Step 2, the end of warmup, reaches the peak 0.1; step 3, the start of the cosine segment, is still 0.1 — this is the normal join between warmup and cosine at the peak.)

## Constraints

- The parameter \(t\) starts at \(0\); each `step()` does \(t\leftarrow t+1\).
- Must match the reference implementation step by step (floating-point comparison `atol=1e-6, rtol=1e-6`, details in notes.md).
- Supports arbitrary `params` (≥1 tensor), and is responsible only for the \(lr\) of the first param_group.
