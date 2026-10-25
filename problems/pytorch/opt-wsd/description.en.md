# Scheduler: WSD Warmup-Stable-Polynomial Decay

## Problem

Cosine annealing needs the total step count from the very beginning and cannot be resumed mid-run. MiniCPM (arXiv:2404.06395) turned the schedule into a three-phase **WSD** form: linear warmup → a **stable** phase (the learning rate stays flat, so training can continue indefinitely and many decay branches can be forked from one checkpoint) → a final **polynomial decay** \(lr\cdot(1-t)^p\) that pushes the rate down to the floor quickly. Wen et al. (arXiv:2410.05192) analyse its loss-drop behaviour on the decay branch. This problem asks you to implement that three-phase scheduler.

You must **subclass `torch.optim.lr_scheduler.LRScheduler`** and return a scheduler instance; the convention is "construction is already step 0, each `step()` call advances one step". The judge reads the learning rate written back to `param_groups[0]["lr"]` at every step and compares the sequence element by element against an independent recomputation (and, for \(p=1\), against `transformers.get_wsd_schedule`).

## Rules

- Factory function `solve(params, base_lr=0.1, warmup_steps=4, stable_steps=6, decay_steps=5, decay_p=1.0, min_lr_ratio=0.0)`: build the optimizer internally with `torch.optim.SGD(params, lr=base_lr)` and return an `LRScheduler` subclass instance over it; `step()` only changes learning rates, never the parameters.
- Let \(f(t)\) be the multiplier of `base_lr` at step \(t\) (0-based), \(lr(t)=\text{base\_lr}\cdot f(t)\), defined piecewise:
  1. **warmup** (`warmup_steps`>0 and \(t<\) `warmup_steps`): \(f=\dfrac{t}{\text{warmup\_steps}}\)
  2. **stable** (\(t<\) `warmup_steps`+`stable_steps`): \(f=1\)
  3. **decay** (`decay_steps`>0 and \(t<\) `warmup_steps`+`stable_steps`+`decay_steps`):
     \[ t_{\text{decay}}=\frac{t-\text{warmup\_steps}-\text{stable\_steps}}{\text{decay\_steps}},\qquad f=(1-t_{\text{decay}})^{\text{decay\_p}} \]
  4. **otherwise** (after the decay phase, and after the stable phase when `decay_steps`≤0): \(f=\) `min_lr_ratio`
- The multiplier is shifted down by the floor `min_lr_ratio` \(m\) and clamped non-negative (this only changes the warmup and decay phases; the other two already satisfy it):
  \[ f\leftarrow\max\big(0,\;f\cdot(1-m)+m\big) \]
- Everything is in plain floats; `decay_p` may be any positive real (1 is linear decay, 2 is squared decay, 0.5 gives a root-shaped branch).
- Calling `transformers.get_wsd_schedule`, `LambdaLR`, `LinearLR` or any ready-made scheduler is **forbidden**; write the piecewise function yourself.

## Examples

With `base_lr=0.5`, `warmup_steps=4`, `stable_steps=6`, `decay_steps=5`, `decay_p=2.0`, `min_lr_ratio=0.1`, reading \(t=0..14\):

```
[0.05, 0.1625, 0.275, 0.3875,      # warmup (starts at 0.1*base after the floor shift)
 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,     # stable phase
 0.5, 0.338, 0.212, 0.122, 0.068]  # (1-t)^2 decay phase and the floor afterwards
```

## Constraints

- All phase lengths are non-negative integers; `decay_steps` may be 0 (the floor is reached right after the stable phase)
- The learning-rate sequence must match the independent recomputation point by point (around \(10^{-9}\)); see notes.md for the cross-check convention
- Beyond `warmup+stable+decay` the rate stays exactly `base_lr·min_lr_ratio`
