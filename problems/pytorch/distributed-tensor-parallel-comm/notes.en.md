# Solution Approach

## Interviewer's Intent

"What is the essential difference between TP and DP communication?" "Why is the TP degree usually capped at 8 GPUs within a single node, while DP can scale to thousands of GPUs?" The key is
the **synchronized object**: DP synchronizes gradients ($\propto$ parameter count, once per step), while TP synchronizes activations ($\propto$ batch·seq·hidden,
**at every layer, on every forward and backward pass**). Once you can compute the per-layer communication bytes of TP, you understand why TP is so sensitive to bandwidth and why it cannot do without NVLink.

## Where the 4 All-Reduces per Layer Come From

Megatron splits a transformer layer into:
- Attention: `QKV` column-parallel → `out_proj` row-parallel → the row-parallel output is a partial sum and needs **1 all-reduce** (forward).
- MLP: `fc1` column-parallel → `fc2` row-parallel → likewise **1 all-reduce** (forward).
- The backward pass is symmetric, adding another 2.

Thus **4 all-reduces per layer**, each with an activation message `act_bytes = batch·seq·hidden·dtype`. On a ring of $W$ GPUs, one all-reduce sends and receives
$2c\cdot\text{act}$ per GPU ($c=\frac{W-1}{W}$), giving $\text{per\_layer}=8c\cdot\text{act}$ in total.

## Why TP Saturates Quickly

As $W$ grows, $c\to1$ and $\text{per\_layer}\to 8\cdot\text{act}$: the communication volume **hardly decreases with $W$** (each GPU still moves roughly a full activation),
while the computation is divided among $W$ GPUs, so the **communication/compute ratio degrades linearly with $W$**—this is why TP hits the bandwidth wall at a small degree and is only done
inside NVLink-connected nodes. Compare with DP: the gradient all-reduce happens once per step, its volume $\propto$ parameter count, and it can be fully amortized by the whole step's computation,
so it can scale to thousands of GPUs.

## Implementation Notes

Compute `act_bytes` first (independent of world; `check_act_bytes_independent_of_world` checks exactly this), then multiply by the ring coefficient and by 4, and finally by the number of layers.
When $W=1$, $c=0$ and the latter two terms are zero. The independent judge reference computes the per-layer volume by **accumulating the 4 all-reduces one by one** (it does not directly write the closed form $8c$).

## Related Problems

`parallel-allreduce` / `distributed-zero-communication` (the same ring coefficient $2c$, but there for gradients), `distributed-gather`, and other
primitives. Together with DP/PP/EP, this problem fills in the **TP** communication accounting among the four parallelism dimensions; Sequence Parallelism replaces some all-reduces with all-gather + reduce-scatter to further save memory/bandwidth, which is an advanced extension (deliberately not modeled in the statement).
