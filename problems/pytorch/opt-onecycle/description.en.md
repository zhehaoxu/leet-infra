# Scheduler: OneCycle LR and Antisymmetric Momentum (OneCycleLR)

## Problem

Monotonically decreasing schedules (cosine, linear) all assume the learning rate starts at its maximum and only falls. The **1cycle** policy of Smith & Topin (arXiv:1708.07120, "Super-Convergence") does the opposite: it **rises** from a small rate to the peak `max_lr` and then **falls** to a minimum far below the starting value, forming a single hump, while the momentum moves **antisymmetrically** - down while lr rises, up while lr falls. The floor is controlled separately by `final_div_factor` and may sit well below the base learning rate; that is precisely the recipe behind "very fast training with large learning rates".

This problem asks you to **subclass `torch.optim.lr_scheduler.LRScheduler`** and implement that scheduler, **writing back both the learning rate and the momentum**. Following the convention "construction is already step 0, each `step()` call advances one step", the judge reads the `lr` and `momentum` sequences from `param_groups[0]` and compares them against the **native `torch.optim.lr_scheduler.OneCycleLR`** (item-wise, \(10^{-12}\)) and against an independent recomputation.

## Rules

- Factory function `solve(params, max_lr=0.1, total_steps=10, pct_start=0.3, div_factor=25.0, final_div_factor=1e4, anneal_strategy="cos", cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, three_phase=False)`: build the optimizer internally as `torch.optim.SGD(params, lr=max_lr/div_factor, momentum=max_momentum)` and return an `LRScheduler` subclass over it; `step()` changes hyperparameters only, never parameter values.
- Let `initial = max_lr / div_factor` and `floor = initial / final_div_factor`. In the two-phase form (`three_phase=False`) the phase upper bounds are
  \[ \text{end}_1=\text{pct\_start}\cdot\text{total\_steps}-1,\qquad \text{end}_2=\text{total\_steps}-1 \]
  lr walks \( \text{initial}\to\text{max\_lr}\to\text{floor} \) and momentum walks \( \text{max\_momentum}\to\text{base\_momentum}\to\text{max\_momentum} \).
  The three-phase form (`three_phase=True`) inserts one more segment: \(\text{initial}\to\text{max\_lr}\to\text{initial}\to\text{floor}\), with the second bound \(2\,\text{pct\_start}\cdot\text{total\_steps}-2\) and the third \(\text{total\_steps}-1\); momentum goes \(\text{max}\to\text{base}\to\text{max}\to\text{max}\).
- Inside phase \(i\) (the first phase starts at \(0.0\)):
  \[ \text{pct}=\frac{t-\text{end}_{i-1}}{\text{end}_i-\text{end}_{i-1}}\in[0,1],\qquad
  \text{anneal}(a,b,\text{pct})=\begin{cases}a+(b-a)\dfrac{1-\cos(\pi\,\text{pct})}{2}, & \texttt{cos}\\[4pt] a+(b-a)\,\text{pct}, & \texttt{linear}\end{cases} \]
  lr uses \(\text{anneal}(a_{lr},b_{lr},\text{pct})\) and momentum uses the **same pct** in \(\text{anneal}(a_m,b_m,\text{pct})\); with `cycle_momentum=False` the momentum is never touched.
- The bounds are **floats** (do not round them); once \(t\) exceeds `total_steps-1` the pct clamps to 1 (the terminal values are kept).
- Calling `OneCycleLR` / `SequentialLR` / `LambdaLR` or any ready-made scheduler is **forbidden**.

## Examples

With `max_lr=1.0`, `total_steps=10`, `pct_start=0.3`, `div_factor=10.0`, `final_div_factor=100.0`, `anneal_strategy="cos"`, `cycle_momentum=True`, `base_momentum=0.8`, `max_momentum=0.95`, the \((lr,\ momentum)\) pairs for \(t=0..9\) are:

```
t=0 (0.1,      0.95 )   t=5 (0.611649, 0.858311)
t=1 (0.55,     0.875)   t=6 (0.389351, 0.891689)
t=2 (1.0,      0.8  )   t=7 (0.189067, 0.921762)
t=3 (0.950534, 0.807427) t=8 (0.050466, 0.942573)
t=4 (0.811933, 0.828238) t=9 (0.001,    0.95 )
```

Step 2 is simultaneously the lr peak (1.0) and the momentum trough (0.8); the final lr \(=0.001\) is far below the initial \(0.1\).

## Constraints

- `pct_start ∈ (0,1)`, `div_factor > 1`, `final_div_factor ≥ 1`, `base_momentum ≤ max_momentum`, `total_steps ≥ 2`
- Must agree with the native `OneCycleLR` step by step, for both lr and momentum (cross-check convention in notes.md)
- Plain floats throughout, no tensor state
