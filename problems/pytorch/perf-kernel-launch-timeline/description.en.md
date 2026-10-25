# Kernel Launch Timeline: Busy/Idle Accounting and CUDA Graph Gains

## Problem

A kernel costs time on two independent timelines: the **CPU side**, which pays a fixed overhead for every
launch, and the **GPU side**, which executes kernels serially on a stream. Once kernels get short the CPU can
no longer push the next kernel into the queue in time and the GPU simply waits - this is **launch-bound**
behaviour, the number-one reason for "mysteriously low GPU utilisation". **CUDA Graphs** capture a whole run of
kernels once and replay it with a single launch, which is exactly the remedy the PyTorch docs recommend.

Given one serial stream of kernels, this problem keeps the busy/idle books and quantifies the graph gain.
Scope split: `operator-fusion` counts the **HBM bytes** fusion saves, `roofline-arithmetic` gives the ceiling of a
**single operator**; this problem accounts for the **launch timeline of a kernel sequence** - how launch overhead
turns into GPU idle time.

## Rules

Let $d_i$ = `durations_us[i]` (GPU time of kernel $i$, in $\mu s$), $L$ = `launch_overhead_us` (CPU cost of one
launch, in $\mu s$), $n$ the number of kernels, and $\textstyle W=\sum_i d_i$ the useful compute time.

- **eager (one launch per kernel)**: the CPU issues launches back-to-back from $t=0$, each costing $L$, so kernel
  $i$ becomes **ready** at $r_i=(i+1)L$ (even the first kernel waits for its own launch). Serial execution on one
  stream:
  $$ e_{-1}=0,\qquad s_i=\max(r_i,\;e_{i-1}),\qquad e_i=s_i+d_i $$
  `eager_wall_us` $=e_{n-1}$.
- **graph (the whole run captured, replayed with one launch)**: the single launch releases every kernel, which
  then run back-to-back:
  $$ \text{graph\_wall}=L+W $$
- `wall_us` = `graph_wall_us` if `use_graph`, else `eager_wall_us`.
- `busy_ratio` $=W/\text{wall\_us}$ (useful share of the selected timeline, always $\le 1$; recorded as `1.0`
  when the denominator is 0).
- `launch_bound_count`: the number of kernels that **idle the GPU waiting for a launch** on the *eager*
  timeline, i.e. the count of $i$ with $r_i>e_{i-1}$. It is a property of the eager timeline and does not depend
  on `use_graph`.
- `speedup $=\text{eager\_wall}/\text{graph\_wall}$ (recorded as `1.0` when the denominator is 0, always $\ge 1$).

## Example

```
Input: durations_us=[30,5,5,20], launch_overhead_us=20, use_graph=False
  r = [20, 40, 60, 80]
  k0: s=max(20,0)=20,  e=50   r0>0      -> idle
  k1: s=max(40,50)=50, e=55   r1<e0     -> no idle
  k2: s=max(60,55)=60, e=65   r2>e1     -> idle
  k3: s=max(80,65)=80, e=100  r3>e2     -> idle
  eager_wall=100, W=60, graph_wall=20+60=80, busy=60/100=0.6, idle kernels=3, speedup=100/80=1.25
Output: (100.0, 80.0, 100.0, 0.6, 3, 1.25)

Input: same but use_graph=True
  wall takes graph_wall=80, busy=60/80=0.75 (launch_bound_count stays on the eager basis = 3)
Output: (100.0, 80.0, 80.0, 0.75, 3, 1.25)
```

## Constraints

- $1\le$ `len(durations_us)` $\le 4096$; `durations_us[i]` $\ge 0$; `launch_overhead_us` $\ge 0$ (all floats, $\mu s$)
- Return a 6-tuple `(float, float, float, float, int, float)`; floats are compared with `atol=1e-5, rtol=1e-5`
- The timeline is defined on **one serial stream** only: multi-stream concurrency, CPU queue depth and the cost
  of capturing the graph are not modelled
- Degenerate conventions: with $n=1$ or $L=0$ eager and graph coincide (`speedup=1.0`); with $L=0$,
  `launch_bound_count=0`; with all $d_i=0$, `eager_wall = n*L` and `graph_wall = L`
