# Scheduler: SGDR Cosine Warm Restarts and the Cycle Ledger

## Problem

A single cosine anneal only gets one shot: the learning rate slides down to the trough and training is over. SGDR (Loshchilov & Hutter, arXiv:1608.03983) makes the cosine **periodic** - at the end of each period the rate does not sit at the floor but **jumps back to the peak** and anneals again (a "warm restart", which repeatedly kicks the optimizer out of sharp minima), while the period lengths grow as \(T_i=T_0\cdot T_{mult}^{\,i}\): explore early, refine late. There is no total step count and training can continue forever; the price is that you must maintain a **cycle ledger** \((T_{cur}, T_i)\) yourself.

Implement this scheduler with semantics **bit-identical** to the native `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`. You must **subclass `torch.optim.lr_scheduler.LRScheduler`** and return a scheduler instance under the convention "construction is already step 0, each `step()` call advances one step"; the judge reads the rate written back to `param_groups[0]["lr"]` at every step and compares the sequence element by element against the native reference and a closed-form recomputation, and separately audits the restart ledger.

## Rules

- Factory function `solve(params, base_lr=0.1, T_0=3, T_mult=2, eta_min=0.0)`: build the optimizer internally with `torch.optim.SGD(params, lr=base_lr)` and return an `LRScheduler` subclass instance over it; `step()` only changes learning rates.
- State: \(T_{cur}\) (steps taken inside the current period) and \(T_i\) (length of that period), initialised to \(T_{cur}=0,\ T_i=T_0\); construction is step 0, so \(lr(0)=\) `base_lr` (the cosine peaks at \(T_{cur}=0\)).
- With \(B\) the base_lr of the group and \(\eta=\) `eta_min`, the rate at step \(t\) is
  \[ lr = \eta + (B-\eta)\cdot\frac{1+\cos\!\big(\pi\, T_{cur}/T_i\big)}{2} \]
- The order inside `step()` **must not be reordered** (advance the ledger first, then write lr):
  1. \(T_{cur}\leftarrow T_{cur}+1\)
  2. If \(T_{cur}\ge T_i\): \(T_{cur}\leftarrow T_{cur}\bmod T_i\) and \(T_i\leftarrow T_i\cdot T_{mult}\) (one restart happened)
  3. Write lr using the formula above
- This order implies three facts that must hold: **restarts land exactly where \(T_{cur}=0\) and lr returns precisely to `base_lr`**; period \(n\) (0-based) has length \(T_0\cdot T_{mult}^n\) so restarts occur at \(t=T_0(T_{mult}^n-1)/(T_{mult}-1)\) (for \(T_{mult}=1\) this is \(t=nT_0\)); and the **trough \(\eta\) where \(\cos=-1\) is never reached**, because that step is zeroed before lr is written.
- `eta_min` is a lower bound, never a value the schedule lands on; the schedule has no end, so arbitrarily many steps keep running without clamping or stopping.
- Calling `CosineAnnealingWarmRestarts`, `CosineAnnealingLR`, `SequentialLR` or any ready-made scheduler is **forbidden**.

## Examples

With `base_lr=0.1`, `T_0=3`, `T_mult=2`, `eta_min=0.0`, reading \(t=0..13\):

```
[0.1,      0.075,    0.025,      # period 0: length 3, T_cur=0,1,2
 0.1,      0.093301, 0.075,      0.05,      0.025,      0.006699,   # period 1: length 6, back to peak at t=3
 0.1,      0.098296, 0.093301,   0.085355,  0.075]      # period 2: length 12, peak again at t=9
```

The restart points are `t = 0, 3, 9, ...` (gaps \(3,6,12=T_0\cdot2^n\)); after 0.025 at \(t=2\) comes 0.1, not 0.

## Constraints

- `T_0` is a positive integer, `T_mult` an integer \(\ge 1\), `eta_min` a non-negative real
- The sequence must match both the native reference and the closed-form recomputation step by step (around \(10^{-12}\)); see notes.md for the cross-check convention
- No cap on the number of periods: reads at step 40 or step 60 must keep growing the periods as \(T_0\cdot T_{mult}^n\)
