# Ring All-Reduce Communication Volume and Latency

## Problem

Data-parallel training needs **All-Reduce** every step to sum and synchronize the gradients on all GPUs. Ring All-Reduce is a common implementation without a
central node: it makes each GPU's transfer volume **almost independent of the number of GPUs**, which is exactly why it can scale to large clusters. This problem computes the
per-GPU communication volume and time.

## Rules

- Implement `solve(message_bytes, world, bandwidth_bps, latency_per_hop=0.0)`, returning the tuple
  `(bytes_per_rank, time_seconds)` (both float).
- Ring All-Reduce has two phases: **Reduce-Scatter** + **All-Gather**, each with $W-1$ steps, and each step each GPU sends
  $\frac{S}{W}$ bytes ($S=$ `message_bytes`, $W=$ `world`). Hence each GPU's total send volume:
  $$ \text{bytes\_per\_rank} = 2 \cdot \frac{W-1}{W} \cdot S $$
  Key point: as $W$ grows, the coefficient $\frac{W-1}{W}\to 1$, so the per-GPU transfer volume tends to $2S$ and **no longer grows linearly with $W$**
  (compare the naive parameter-server scheme, where $2S$ all lands on the server).
- **Time** = transfer time + per-hop latency:
  $$ \text{time} = \frac{\text{bytes\_per\_rank}}{\text{bandwidth\_bps}} + 2(W-1)\cdot \text{latency\_per\_hop} $$
  The two phases total $2(W-1)$ communication hops, each with a fixed latency (start-up / synchronization overhead).
- Boundary: for $W \le 1$ **no communication is needed**; return `(0.0, 0.0)`.

## Examples

```
Input: message_bytes=1e9, world=8, bandwidth_bps=1e11 (100GB/s), latency_per_hop=0.0
  bytes_per_rank = 2 * (8-1)/8 * 1e9 = 1.75e9   (1,750,000,000)
  time = 1.75e9 / 1e11 + 2*7*0 = 0.0175 s
Output: (1750000000.0, 0.0175)
```

## Constraints

- `message_bytes ≥ 0`, `world ≥ 1` (integer), `bandwidth_bps > 0`, `latency_per_hop ≥ 0`
- Returns the tuple `(bytes_per_rank, time_seconds)`, both float
- The grader uses a relative tolerance of `1e-6` for floats
