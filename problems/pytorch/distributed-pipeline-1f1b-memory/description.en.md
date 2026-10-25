# Peak Activation Memory of 1F1B Pipeline Parallelism

## Problem

`distributed-pipeline-bubble` computes the **time (bubbles)** of pipeline parallelism; this problem computes its other account — **activation memory**. To fill the pipeline, GPipe runs all micro-batches' forward passes first, at the cost of holding all $m$ copies of activations in each stage's memory. The **1F1B** (one-forward-one-backward) schedule interleaves forward and backward, **capping** the number of micro-batches in flight at each stage and greatly reducing activation memory. This problem computes that peak (pure computation, with a simplified model defined here).

## Model

The model's $L$ layers are split evenly across $p$ stages, $\text{lps} = L/p$ (divisibility required). Let `act_bytes` = the bytes of activation that **one micro-batch at a single layer** must save for the backward pass; then one in-flight micro-batch occupies, **at a given stage** (including its $\text{lps}$ layers),
$$ \text{per\_mb} = \text{lps}\cdot \texttt{act\_bytes} $$
A stage's peak memory = `per_mb` × "the number of micro-batches in flight at that stage":

- **GPipe**: backward starts only after all $m$ forward passes finish, so each stage stores $m$ copies
  $$ \text{gpipe\_peak} = \text{lps}\cdot \texttt{act\_bytes}\cdot m $$
- **1F1B**: segment `i` (counted from the input end, $i=0$) holds at most $\min(m,\; p-i)$ in-flight activations during warmup. Hence
  - heaviest segment $i=0$: $\text{offfb\_worst} = \text{lps}\cdot \texttt{act\_bytes}\cdot \min(m, p)$
  - lightest segment $i=p-1$: $\text{offfb\_best} = \text{lps}\cdot \texttt{act\_bytes}\cdot \min(m, 1)$

## Rules

Implement `solve(n_layers, n_stages, n_microbatches, act_bytes)` and return the triple
`(gpipe_peak, offfb_worst, offfb_best)` (float byte counts).

## Examples

```
L=32, p=4, m=8, act_bytes=1e6   →  lps=8, per_mb=8e6
gpipe_peak = 8e6·8        = 6.4e7
offfb_worst = 8e6·min(8,4) = 3.2e7      # stage0 stores half as much
offfb_best  = 8e6·min(8,1) = 8.0e6      # the last segment stores only 1 copy
→ (6.4e7, 3.2e7, 8.0e6)
```

## Constraints

- `1 ≤ n_stages ≤ n_layers` (divisibility only makes sense then), `1 ≤ n_microbatches`; `act_bytes ≥ 0`
- **`n_layers` must be divisible by `n_stages`**
- Return `(float, float, float)`, compared with relative error $10^{-6}$
- Key point: when $m\le p$, `offfb_worst == gpipe_peak` (1F1B saves nothing at the heaviest segment); savings occur only when $m>p$, where the heaviest-segment ratio is $p/m$
- Only activation memory is computed; parameters/gradients/optimizer states are ignored (see `training-state-memory` for those)
