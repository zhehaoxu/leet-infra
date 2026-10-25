# In Depth: The Checkpoint I/O Bill, Asynchronous Saves and the Optimal Interval

## Source

- **CheckFreq** (Chen et al., arXiv:2301.13327) writes the cost of saving versus the cost of redoing work after a
  failure as one optimisable curve; its closed form goes back to the classic Young (1974) / Daly (2006) result:
  optimal interval $\approx\sqrt{2CF}$, with $C$ the time one checkpoint **blocks** the loop and $F$ the mean time
  between failures.
- **Asynchronous saving** is Megatron-LM's `--async-save` (copy the state into pinned host staging, then let a
  background thread write it out) and PyTorch's `torch.distributed.checkpoint.async_save`.
- The state byte count reuses this bank's `training-state-memory` convention (mixed precision + Adam = 16 bytes per
  parameter).

## Key Points and Pitfalls

1. **Divide by each bandwidth separately, do not merge them**: `copy_ms` uses HBM→host (PCIe/NVLink, tens of GB/s)
   while `write_ms` uses host→storage (parallel filesystem / NVMe, often only 1-10 GB/s). The whole point of
   asynchrony is to take the **slow** stage off the critical path: the stall shrinks from `copy+write` to `copy`.
2. **`period_ms = checkpoint_interval * step_ms`**: the overhead fraction is `stall/period`, not `stall/step_ms`.
   The two coincide only when a checkpoint happens every step, which is exactly the degenerate case the grader uses
   to pin the convention.
3. **`overhead` may exceed 1**: that means saving is slower than training and the configuration is simply
   infeasible - do not clamp it.
4. **The slot is held for copy+write, not just write**: a pinned buffer stays occupied from the start of the copy
   until the write finishes, so the can't-keep-up test is `copy+write > B*period` (**strictly** greater; equality
   still rotates cleanly). Using `write > B*period`, or treating `>=` as the limit, fails on the boundary cases.
5. **$C$ in Daly's formula is the blocking cost only**: $C=\text{copy}/\text{step}$ when async,
   $C=(\text{copy}+\text{write})/\text{step}$ when sync. Charging the overlapped `write` to $C$ would (absurdly)
   suggest that async saves should be *rarer*.
6. **Keep units straight**: bandwidths are bytes per **second**, so times need a factor 1000 to become ms; $C$ is
   "how many **steps** are blocked", so divide by `step_ms`, never multiply.
7. **Floor, then clamp to 1**: `floor`, not `round`; a result below 1 means "even saving every step is sparser than
   the formula asks", and by definition becomes 1.
8. **Division of labour**: the read side (prefetching parameters into HBM during training) belongs to
   `distributed-zero-offload`, the recompute side to `distributed-sublinear-checkpoint` / `transformer-mem`; this
   problem is only the write-side I/O bill.

## Cross-check Approach

The reference solution uses closed forms: `state = n_params*bps`, `stall = copy (+write)`,
`writer_limited = copy+write > B*period`, `optimal = floor(sqrt(2*(stall/step)*mtbf))`. The grader's `_oracle`
re-derives all four by different routes: state is split into thousand-element blocks and recombined
(`n//1000*(1000*bps) + n%1000*bps`); stall is "sync total minus the write that async hides"; `writer_limited` comes
from a **60-event simulation of the B staging slots** (any checkpoint that must wait for a slot means the writer is
limited); and `optimal` puts `step_ms` inside the root (`sqrt(2*stall*mtbf/step)`). Degenerate assertions cover
`stall_sync - stall_async == write_ms`, `overhead == stall/step_ms` at interval 1, a strict-boundary table over
(B, period), the infinite-bandwidth case (all zeros, interval clamped to 1), and halving of every time when the
state bytes are halved.

## Complexity

The reference solution is $O(1)$. The grader's slot simulation is $O(60\cdot B)$, i.e. constant, so the whole suite
runs in milliseconds to tens of milliseconds.
