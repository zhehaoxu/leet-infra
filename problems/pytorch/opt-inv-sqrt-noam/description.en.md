# Scheduler: Noam Inverse Square Root Learning Rate

## Problem

The original Transformer paper (Vaswani et al., arXiv:1706.03762, equation (5)) defines its schedule with only two quantities - the model width \(d_{model}\) and the warmup length \(w\):

\[ lr(t)=d_{model}^{-1/2}\cdot\min\!\big(t^{-1/2},\; t\cdot w^{-3/2}\big) \]

The first branch decays like an inverse square root, the second climbs linearly, and they cross at \(t=w\), so the curve is "crawl up to the peak, then a long \(t^{-1/2}\) tail down". It has **no total step count** and **no floor clamping**, so training can be resumed from any step; it is the prototype of the "you do not need to know the training length" family (AdaFactor's relative step size and Big Vision's inverse-sqrt are both rewrites of it).

Implement this scheduler. You must **subclass `torch.optim.lr_scheduler.LRScheduler`** and return a scheduler instance under the convention "construction is already step 0, each `step()` call advances one step"; the judge reads the rate written back to `param_groups[0]["lr"]` at every step and compares the sequence element by element against an independent recomputation (the paper's min closed form) and against `transformers.get_inverse_sqrt_schedule`.

## Rules

- Factory function `solve(params, model_dim=512, warmup_steps=4000)`: build the optimizer internally with `torch.optim.SGD(params, lr=model_dim**-0.5)` and return an `LRScheduler` subclass instance over it; `step()` only changes learning rates and never touches the parameters.
- Let \(t\) be the current step (0-based), \(d=\) `model_dim`, \(w=\) `warmup_steps`. Use the piecewise form (equivalent to the formula above for \(t\ge1\), but at \(t=0\) the linear branch must give 0 - you may **not** actually evaluate \(0^{-1/2}\)):
  \[ lr(t)=\begin{cases} d^{-1/2}\cdot t\cdot w^{-3/2}, & t<w\\[4pt] d^{-1/2}\cdot t^{-1/2}, & t\ge w \end{cases} \]
- Hence \(lr(0)=0\), the peak is \(lr(w)=d^{-1/2}w^{-1/2}\), and the curve is continuous at \(t=w\) (the increment there still equals the warmup slope \(d^{-1/2}w^{-3/2}\)).
- The rate is an **absolute** value, not a multiplier of each group's base_lr; with several parameter groups all of them get the same \(lr(t)\).
- No total step count and no `min_lr`: larger \(t\) only keeps decaying as \(t^{-1/2}\), always positive, never clamped, never stops.
- `warmup_steps` is a positive integer (1 is allowed: the peak is reached at step 1 and everything after is pure decay); `model_dim` is a positive integer.
- Calling `get_inverse_sqrt_schedule`, `LambdaLR`, `LinearLR` or any ready-made scheduler is **forbidden**; write the piecewise function yourself.

## Examples

With `model_dim=64`, `warmup_steps=5` (peak \(=64^{-1/2}\cdot5^{-1/2}\approx0.0559017\)), reading \(t=0..12\):

```
[0.0, 0.01118034, 0.02236068, 0.03354102, 0.04472136,   # linear warmup, common difference d^-0.5*w^-1.5
 0.0559017,                                             # the peak at t=w=5
 0.05103104, 0.04724556, 0.04419417, 0.04166667,        # the t^-0.5 long tail
 0.03952847, 0.03768892, 0.03608439]
```

## Constraints

- All in floats; for \(0\le t\le 60\) the sequence must match the paper's min closed form point by point (around \(10^{-12}\)) and agree with `transformers.get_inverse_sqrt_schedule` (using `timescale=warmup_steps` and the optimizer lr set to the peak) within a relative error of \(10^{-12}\)
- Structural invariants: arithmetic warmup, the peak exactly at \(t=\) `warmup_steps`, strict monotone decrease afterwards and always positive
- `warmup_steps>=1`, `model_dim>=1`; evaluating \(t=0\) must not raise a division by zero nor return `inf`/`nan`
