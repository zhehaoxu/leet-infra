# Notes · Placement Accounting for Disaggregated Prefill/Decode

## Sources

- **Mooncake** (arXiv:2407.00079, Moonshot AI): a KVCache-centric disaggregated architecture — separate prefill and decode pools, with KV moved from prefill nodes to decode nodes over RDMA and KV cache treated as a first-class resource for placement and scheduling. The chain "prefill done tick → KV transfer → resident on decode" follows this shape.
- **DistServe** (arXiv:2401.09670): after decoupling prefill from decode, placement is judged by goodput under TTFT/TPOT SLOs; this exercise abstracts "KV transfer plus memory admission decides whether a request is admitted" into a deterministic ledger.
- The ring-exchange share matches `distributed-sequence-parallel` in this repository: for an all-gather over C ranks, each link carries only a `(C-1)/C` share. Here the intra-node TP exchange is also charged to the same link — a **simplification**.

## Core points and pitfalls

**Ceiling on the cumulative total.** `ceil(cum/tps)` differs from "ceil each request, then sum" whenever tps does not divide the lengths (example: three prompts of length 3 with tps=2 gives 2/3/5 ticks, while the per-request reading gives 2/4/6). The per-request reading makes TTFT grow linearly in the number of requests.

**Dropped requests still consumed prefill.** A request that fails admission occupies neither memory nor link, but the time it spent in the prefill queue was real; later requests' `prefill_done` must keep counting it. Removing dropped requests from the cumulative total would pull every later TTFT earlier.

**The ring share is per link, not total traffic.** `moved = kv + ceil(kv*(C-1)/C)`: nothing extra at tp=1, half again at tp=2, three quarters at tp=4. Do not write `kv*C` (that is broadcast total traffic) or `kv*(C-1)/C` (that drops the handoff itself).

**Capacity is the peak.** `kv = (prompt + gen) * bytes_per_token`; KV keeps growing during generation, so counting only the prompt over-admits. Admission is `resident + kv <= caps`, with equality still fitting.

**Place by predicted completion, not by idleness.** A replica that is free earlier but has a slow link can lose to one that is free later with a wider link; `check_earliest_finish_wins` is built for exactly this. Ties fall back to the smaller index.

**Estimate before committing.** All candidate `start` values must use the `link_free` state **before** this request is committed; updating as you scan corrupts the estimates of later candidates.

**TTFT is defined as KV-resident**, excluding the first decode step. Compare everything on that tick; do not add an extra constant.

## Related problems

- `inf-cache-aware-routing`: picks instances by prefix-cache hit; this problem has no cache, only transfer and admission.
- `inf-kv-residency-billing`: block·step integral and billing; `inf-paged-cow-block-share`: block-level COW; `inf-kv-block-admission`: block-pool admission. All three manage blocks within one instance and never move KV across nodes.
- `inf-preempt-swap-recompute`: preempts running requests when memory is short and compares swap vs recompute; here a **new** request is placed or dropped, with no preemption.
- `distributed-sequence-parallel`: only the single memory-communication formula `2*f*A*(C-1)/C`; this problem reuses its ring coefficient but lands it on multiple replicas, queues, and admission.
