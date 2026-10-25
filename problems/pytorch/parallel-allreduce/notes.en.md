# Solution Approach

## Interviewer's intent

"Adding GPUs to data parallelism runs into a communication bottleneck — why? Why use Ring All-Reduce instead of a central parameter server?" The
standard answer to such questions is this problem's $2\cdot\frac{W-1}{W}S$. It is a cornerstone of distributed-training communication analysis, and is often asked together with `training-state-memory`
and ZeRO.

## Why it is $2\cdot\frac{W-1}{W}S$

Ring connects $W$ GPUs into a ring and splits All-Reduce into two halves:

1. **Reduce-Scatter**: $W-1$ steps; each step each GPU sends $\frac1W$ of the data to the next one and accumulates, and at the end each GPU holds
   $\frac1W$ of the **complete sum result**. Send volume $=(W-1)\cdot\frac{S}{W}$.
2. **All-Gather**: another $W-1$ steps forward these partial results around the ring so that each GPU gets **all** $\frac WW$ segments. The same volume.

Add the two segments: $2\cdot\frac{W-1}{W}\cdot S$.

## Key insight: nearly independent of the number of GPUs

The coefficient $\frac{W-1}{W}$ approaches $1$ quickly as $W$ grows, so the per-GPU transfer volume is capped at $2S$ and **does not blow up linearly with the number of GPUs**.
Compare the naive scheme: a parameter server must have the central node send and receive $2S\cdot W$, so the more GPUs there are, the more congested the server is. This is the fundamental reason Ring can scale
to hundreds or thousands of GPUs (when the bandwidth term dominates).

## The role of the latency term

The $2(W-1)$ hops each carry a fixed start-up latency; for small messages / very large clusters the latency term $\propto W$ overtakes the bandwidth term and becomes the bottleneck —
which is also why in practice **double binary tree / halving-doubling** replace pure ring. This problem provides the `latency_per_hop`
parameter so you can include this term (the default of 0 considers bandwidth only).

## Common mistakes

- Missing the factor 2 (computing only reduce-scatter or only all-gather).
- Writing the per-GPU transfer volume as $2S\cdot(W-1)$ (missing the division by $W$), which degenerates into the server's linear growth.
- Not special-casing $W=1$, returning a nonzero communication volume.

## Grading details

`test_cases.py` checks the closed form against an independent reference that **accumulates in two phases**, and fixes three invariants: $W=1$ zero communication, per-GPU transfer volume
$\le 2S$ and monotonically approaching $2S$ as $W$ grows, and the latency term only added to time without changing the volume.
