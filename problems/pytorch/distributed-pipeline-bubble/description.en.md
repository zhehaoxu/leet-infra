# Pipeline Parallelism Bubble Ratio (GPipe)

## Problem

Pipeline Parallelism (PP) splits the model by layer into $p$ stages assigned to $p$ GPUs, and a batch is further split into $m$ **micro-batches** that flow through the stages in turn to fill the pipeline. But during warmup and drain, some GPUs are always idle — these are the **bubbles**. The bubble fraction directly determines the efficiency of pipeline parallelism and is the yardstick interviewers use to gauge whether you "understand GPipe/1F1B". This problem computes it (pure computation).

## Model (GPipe: all forwards, then all backwards)

Let one micro-batch passing through **one stage** count as 1 time unit. With $m$ micro-batches and a $p$-stage pipeline:

- **makespan (total wall-clock time, in units)**: from the first stage starting to the last stage finishing
  $$ T = m + p - 1 $$
- **useful work (per stage)**: the time each stage is actually busy $= m$ (processing $m$ micro-batches)
- **utilization**: $\text{util} = \dfrac{m}{m+p-1}$
- **bubble ratio**: the idle fraction
  $$ \text{bubble} = \frac{T - m}{T} = \frac{p-1}{m+p-1} = 1 - \text{util} $$

(With symmetric fwd/bwd, the forward and backward times cancel in the ratio, so it depends only on $m,p$ — this is exactly the classic GPipe result.)

## Rules

Implement `solve(n_stages, n_microbatches)` (let $p=$`n_stages`, $m=$`n_microbatches`) and return the triple
`(bubble, util, makespan)`: `bubble` and `util` are floats, `makespan` is a float ($=m+p-1$).

## Examples

```
p=4, m=8:  makespan=11, util=8/11=0.727272…, bubble=3/11=0.272727…
→ (0.2727272727, 0.7272727273, 11.0)
```

## Constraints

- `1 ≤ n_stages, n_microbatches ≤ 4096`
- $p=1$: no pipeline, bubble $=0$, util $=1$, makespan $=m$
- $m=1$: no overlap possible, bubble $=(p-1)/p$, util $=1/p$, makespan $=p$
- Return `(float, float, float)`, compared within relative error $10^{-6}$; must satisfy `bubble + util == 1`
- Only GPipe bubbles are computed; the specific 1F1B schedule or memory is not involved
