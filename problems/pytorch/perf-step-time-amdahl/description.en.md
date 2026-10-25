# Training Step Time Breakdown and the Amdahl Ceiling

## Problem

A profiler's view of one training step is usually four serial phases: **dataloader wait / forward / backward /
optimizer update**. From that table you must answer two questions: (1) "if I make one phase k times faster, how much
faster does the whole step get?" - Amdahl's law, whose ceiling is fixed by the share you did *not* optimise; and
(2) "can the data wait be hidden inside the compute using `num_workers` + `prefetch`?" - the input-pipeline account.
This problem turns both into pure arithmetic. `transformer-mfu` accounts for the step's compute utilisation and
`perf-kernel-launch-timeline` for GPU-side launch idleness; this problem covers **phase attribution + speed-up
ceiling + data hiding**.

## Rules

`phase_ms` holds 4 non-negative numbers ordered as `[dataloader_wait, forward, backward, optimizer]` (milliseconds,
serial-phase convention). Let `total` $=\sum_i \texttt{phase\_ms}[i]$, $p=$ `phase_ms[target_phase]`, $k=$ `speedup`.

- **Shares**: $\text{fracs}[i]=\texttt{phase\_ms}[i]/\text{total}$ (a 4-element `float64` tensor summing to 1).
- **Amdahl**: only the named phase is compressed, everything else is kept as is:
  $$ \text{new\_step}=(\text{total}-p)+\frac{p}{k},\qquad
     \text{amdahl}=\frac{\text{total}}{\text{new\_step}},\qquad
     \text{ceiling}=\lim_{k\to\infty}\text{amdahl}=\frac{\text{total}}{\text{total}-p} $$
  `ceiling` is `inf` when `total == p` (all other phases are zero); `amdahl` is `inf` when `new_step == 0`.
- **Data hiding** (a separate view from `total` - do not mix them): with $W=$ `num_workers`, the supply interval is
  $$ \text{supply}=\frac{\texttt{worker\_ms}}{W},\qquad \text{compute}=\texttt{phase\_ms}[1]+\texttt{phase\_ms}[2]+\texttt{phase\_ms}[3] $$
  `prefetch` $=0$ means synchronous loading with no overlap at all: $\text{wait}=\text{supply}$; with
  `prefetch` $\ge 1$ the queue can hold work ahead, so in steady state you only pay the shortfall:
  $$ \text{wait}=\max\!\big(0,\ \text{supply}-\text{compute}\big),\qquad \text{effective}=\text{compute}+\text{wait} $$
  `data_hidden` $=(\text{wait}=0)$ - supply **exactly** equal to compute counts as hidden.
  `effective` is the **steady-state amortised** step time; the warm-up of the first batch is not charged.

## Examples

```
Input: phase_ms=[20,100,200,40], target_phase=0, speedup=4, worker_ms=80, num_workers=4, prefetch=2
  total=360; fracs=[20/360, 100/360, 200/360, 40/360]
  new_step = (360-20) + 20/4 = 345      amdahl = 360/345 = 1.0435
  ceiling  = 360/340 = 1.0588            <- even driving the dataloader to zero buys only 5.9%
  supply = 80/4 = 20; compute = 340; 20 <= 340 -> wait=0 -> effective=340, hidden=True
Output: (tensor([0.0555…, 0.2777…, 0.5555…, 0.1111…]), 360.0, 345.0, 1.0435…, 1.0588…, 340.0, True)

Input: phase_ms=[150,60,120,20], target_phase=0, speedup=8, worker_ms=90, num_workers=3, prefetch=0
  total=350; new_step = 200 + 150/8 = 218.75; amdahl = 350/218.75 = 1.6; ceiling = 350/200 = 1.75
  supply = 90/3 = 30; compute = 200; prefetch=0 -> wait=supply=30 -> effective=230, hidden=False
Output: (tensor([0.4285…, 0.1714…, 0.3428…, 0.0571…]), 350.0, 218.75, 1.6, 1.75, 230.0, False)
```

## Constraints

- `phase_ms` has exactly 4 non-negative numbers with `total > 0`; `target_phase` $\in\{0,1,2,3\}$; `speedup` $>0$
- `worker_ms` $\ge 0$; `num_workers` $\ge 1$; `prefetch` $\ge 0$ (integers)
- Return `(fracs, total_ms, new_step_ms, amdahl_speedup, ceiling_speedup, effective_step_ms, data_hidden)`:
  item 1 is a `float64` tensor, items 2-6 are floats compared with `atol=1e-5, rtol=1e-5`, item 7 is a bool
- Phases add up serially (no multi-stream overlap); `fracs` uses the measured `phase_ms[0]`, not the worker model
