# Strong/Weak Scaling: Speedup, Efficiency and the Saturation Point

## Problem

"Do 8 GPUs mean 8× faster?" A scaling curve answers two things: the **speedup** - how much faster than the
baseline - and the **efficiency** - the speedup divided by the GPU multiplier, ideally 1. The two scaling styles
use different yardsticks: **strong scaling** keeps the total work fixed (more GPUs, less work per GPU),
**weak scaling** keeps the per-GPU work fixed (more GPUs, more total work). Efficiency $>1$ is **superlinear**
(often because sharding collapses the per-GPU communication/memory pressure), and the first configuration where
throughput stops rising is the **saturation point** - past it, extra GPUs only burn money.

`distributed-pipeline-bubble` and `distributed-zero-communication` explain *why* efficiency drops (bubbles,
communication); this problem is about *how to compute* the curve.

## Rules

`world_sizes` is an ascending list of positive integers $w[i]$ (GPUs per configuration), `step_times_ms` an
equal-length list of positive numbers $t[i]$ (wall-clock **time per step** of that configuration, ms). Index 0 is
the baseline.

- **Aggregate throughput (relative)**:
  - `mode == "strong"`: the number of samples per step is fixed $\Rightarrow$ throughput $\propto 1/t_i$
  - `mode == "weak"`: samples per GPU are fixed $\Rightarrow$ samples per step $\propto w_i$, throughput $\propto w_i/t_i$
  $$ \text{speedup}_i=\frac{\text{throughput}_i}{\text{throughput}_0},\qquad
     \text{efficiency}_i=\frac{\text{speedup}_i}{w_i/w_0} $$
  Equivalent closed forms: strong $\text{speedup}=t_0/t_i$, $\text{efficiency}=(t_0/t_i)(w_0/w_i)$;
  weak $\text{speedup}=(w_i t_0)/(w_0 t_i)$, $\text{efficiency}=t_0/t_i$.
  In both modes $\text{speedup}_i=\text{efficiency}_i\cdot(w_i/w_0)$ always holds, and
  $\text{speedup}_0=\text{efficiency}_0=1$.
- `superlinear`: the indices with $\text{efficiency}_i>1.0$ (**strictly** greater; efficiency exactly 1 does not
  count), as an ascending `int64` tensor.
- `saturation_index`: the smallest $i\ge1$ with $\text{throughput}_i\le\text{throughput}_{i-1}$ (adding GPUs no
  longer raises aggregate throughput; ties count), or `-1` if the curve rises strictly.

## Examples

```
Input: world_sizes=[1,2,4,8,16], step_times_ms=[100,45,25,15,9], mode="strong"
  speedup    = [1, 100/45, 100/25, 100/15, 100/9] = [1.0, 2.2222…, 4.0, 6.6667…, 11.1111…]
  efficiency = speedup / w = [1.0, 1.1111…, 1.0, 0.8333…, 0.6944…]
  superlinear=[1] (only the 2-GPU tier beats 1; the 4-GPU tier is exactly 1.0); throughput rises -> saturation=-1
Output: (tensor([1.0, 2.2222…, 4.0, 6.6667…, 11.1111…]),
        tensor([1.0, 1.1111…, 1.0, 0.8333…, 0.6944…]), tensor([1]), -1)

Input: world_sizes=[1,2,4,8], step_times_ms=[100,105,115,240], mode="weak"
  efficiency = t0/t = [1.0, 0.9524…, 0.8696…, 0.4167…]
  speedup    = efficiency * w = [1.0, 1.9048…, 3.4783…, 3.3333…]
  8 GPUs move 8/240 per ms, below 4 GPUs at 4/115 -> saturation=3; no superlinear point
Output: (tensor([1.0, 1.9048…, 3.4783…, 3.3333…]),
        tensor([1.0, 0.9524…, 0.8696…, 0.4167…]), tensor([]), 3)
```

## Constraints

- List length $1\ldots 1024$; both lists have the same length; `world_sizes` are strictly ascending positive
  integers; `step_times_ms` are positive numbers
- `mode` is `"strong"` or `"weak"`
- Return `(speedup, efficiency, superlinear, saturation_index)`: the first two are `float64` tensors compared with
  `atol=1e-5, rtol=1e-5`, the third an `int64` tensor (`torch.equal`), the fourth an `int`
- Only the throughput-ratio convention is modelled - no bandwidth or topology; the saturation test includes ties,
  the superlinear test excludes efficiency exactly 1
