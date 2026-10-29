# Deep Dive: The Offload Prefetch Pipeline and Compute-Transfer Overlap Accounting

## Origin
ZeRO-Offload (2101.06840) keeps optimizer states and gradients on the CPU and streams them layer by layer; ZeRO-Infinity (2104.07857) pushes parameters and KV further out to NVMe and hides the latency with **prefetching**. Both rely on the same pipeline: "overlap compute with copies over a small set of staging buffers". This problem abstracts that pipeline into a single serial engine with `buffers` slots on a deterministic timeline — no real DMA preemption, no multi-stream concurrency, just "when can the prefetch start, and when does compute get dragged". Contrast with `distributed-zero-offload` / `distributed-zero-infinity`, which compute the **capacity and traffic ledger** (how many bytes, how much memory saved), and with `perf-kernel-launch-timeline`, which covers kernel launches and stream dependencies rather than DRAM→HBM staging slots.

## Key points and pitfalls
1. **A slot is held from prefetch start until compute end**, not just until the transfer finishes. Once copied, the data sits in the slot waiting to be consumed, and the slot cannot be reused meanwhile. Miss this and the third layer with `buffers=2` starts copying too early, yielding an optimistic makespan.
2. **Two constraints, take the max each time**: prefetch start is bounded by "engine idle" and "the previous occupant of the same slot finished computing"; compute start is bounded by "previous layer finished computing" and "own prefetch finished". Dropping either gives either compute stalling on data or two copies colliding.
3. **Transfer time rounds up**: `ceil(bytes/bw)`; 40 bytes at 30 bytes/unit costs 2 ticks. Integer division makes transfers too fast and the makespan too small.
4. **`buffers=1` is the degenerate anchor**: no overlap at all, so `stall` equals the total transfer time and `makespan = Σcompute + Σtransfer`. Failing this identity means the slot logic is wrong.
5. **Zero-byte layers still obey slot rules**: they cost no engine time, but with `buffers=1` they still cannot be "issued" until the previous layer finishes computing, so their `transfer_start` is not 0 — a case the tests pin down.
6. **Stall = makespan − Σcompute**: compute accumulates serially, so every gap comes from a prefetch falling behind. The two definitions (sum of `c_j - finish[j-1]` vs total minus compute) must agree.

## Verification
The reference advances three timestamps per layer (`engine_free`, `finish[j-buffers]`, `finish[j-1]`), one formula per layer. The judge `_oracle` ignores formulas and runs a **unit-time tick** state machine: the engine moves `bw` bytes per tick, slots are tracked with an occupied flag, compute and transfer run in parallel within a tick, and zero-byte layers chain within the same tick; the stall emerges as "makespan − Σcompute". The two structures are compared field by field over 30 random cases. Additional assertions: with a single slot the stall equals the total transfer time; zero-byte layers' `transfer_start` still obeys slot rules; `makespan = Σcompute + stall`; every layer has `c_j >= e_j` and compute never overlaps; `transfer_start[j] >= finish[j-buffers]`; adding slots never increases the makespan; when compute dwarfs transfer only the first prefetch stalls; and the empty input degenerates.

## Complexity
`O(L)` — three max operations per layer. The tick oracle is `O(makespan)`, on very small inputs.
