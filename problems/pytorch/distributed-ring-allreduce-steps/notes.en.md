# Approach

## Interviewer's Intent

The gap between someone who only calls `dist.all_reduce` and someone who can explain "on the ring there are $2(W-1)$ steps, each moving one chunk, and why the communication volume is nearly independent of $W$" is exactly here. This problem breaks Ring All-Reduce apart so you can run it by hand — the only hardcore way to understand NCCL's bandwidth optimality.

## Why These Two Phases

All-Reduce = **Reduce-Scatter** $+$ **All-Gather**.

- Reduce-Scatter: $W-1$ steps flow "the sum of each chunk across ranks" around the ring; when done, rank `q` fully owns only chunk `q` (the diagonal).
- All-Gather: another $W-1$ steps spread these "complete single chunks" around the ring; when done, everyone holds the complete sum of all chunks.

Each rank sends only **one chunk** per step (size $S/W$), for $2(W-1)$ steps, so the **communication volume per GPU** is $=2(W-1)\cdot\frac SW$ $=2\frac{W-1}{W}S$ — matching the ring coefficient $2cS$ in `parallel-allreduce` and `distributed-zero-communication` exactly. The key insight: this quantity barely grows with $W$ ($c\to1$) and is bandwidth-optimal, which is exactly why ring beats "everyone sends to rank0 and it broadcasts".

## Two Pitfalls of the Schedule

1. **Synchronous round semantics**: within a step, all sends must be based on the buffer **before** the update. Compute all of the step's `sends` first, then apply them to `bufs` together; otherwise values modified within the same step are used again by downstream ranks and the intermediate states are wrong.
2. **Index formulas**: Reduce-Scatter sends $(q-1-t)\bmod W$, All-Gather sends $(q-t)\bmod W$; they differ by one offset — being off by one makes a "partial" value get spread as if it were the complete sum. The statement gives the full trace for $W=3$; if your code is wrong, hand-compute and compare `step0/step1`.

## Implementation Notes

Start `bufs` from `.clone()` of the input rows (do not modify `init` in place); each step store a $(W,W)$ snapshot with `torch.stack`. For $W=1$ both `range(0)` loops do not execute and `[]` is returned naturally. Grading cross-checks with an independent replay `_ref` of the same schedule, and additionally verifies that the last snapshot == the All-Reduce result, that the diagonal after reduce-scatter == `full`, that the length is $=2(W-1)$, and that the input is not modified.

## Related Problems

`distributed-all-reduce` (final value), `distributed-reduce-scatter` / `distributed-all-gather` (single-phase semantics), `parallel-allreduce` (converting the step count here into communication bytes and elapsed time).
