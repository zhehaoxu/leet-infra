# Solution Notes · Preemption: Swap-Out vs Recompute

## Origin

- vLLM's scheduler implements "out of memory → preempt" with two paths: `RECOMPUTE` (drop the KV and re-prefill later, the default) and `SWAP` (move the KV to CPU and bring it back). Victims come from the tail of the running queue, chosen by request priority. See `vllm/core/scheduler.py` (`_preempt`, `PreemptionMode`).
- Orca (OSDI '22) iteration-level scheduling and Sarathi-Serve (arXiv:2403.02310) chunked prefill form the background of "how a preempted request resumes": recomputed requests pay the prefill compute again.
- The quantitative swap-vs-recompute comparison is an abstraction of those two implementation paths: transfer cost scales with KV bytes, recompute cost with the whole sequence length.

## Key points and pitfalls

**Recompute pays for `prompt + generated`, not just `prompt`.** The tokens already generated must be re-run to return to the same state; dropping `generated` wrongly makes long-output requests look cheap to recompute.

**Swap transfers twice.** Out now, back in later: `2 × kv × bytes_per_token / bandwidth`. Counting one way halves swap's apparent cost.

**The ceiling must be integer.** Use `-(-a // b)` or `(a + b - 1) // b`; a float division can round the wrong way when the quotient sits just above an integer.

**Tiers first, then size.** The order is "priority ascending → larger kv → smaller index" lexicographically, not a global sort by kv. In Example 2 the `kv = 6` request goes before `kv = 50` because it sits in the lower priority tier.

**Stop when enough.** `freed` may exceed `need` (80 > 44 in Example 1), but never preempt one extra request "for good measure".

**Infeasibility gives up entirely.** When every candidate is exhausted and `need` is still unmet: `victims`/`modes` empty, `ticks = -1`, `replay_tokens = 0`, while `freed` still reports what could have been freed — that field is the diagnostics for the caller, so do not zero it.

**Ties go to swap.** `swap_ticks <= recompute_ticks` selects swap; the boundary case (swap exactly equal to recompute) must take the swap branch.

## Verification

- Reference: `sorted({...})` builds the priority tiers, `candidates.sort(key=(-kv, index))` orders within a tier, and the main loop short-circuits with two `if freed >= need: break` checks.
- Independent oracle: buckets into a `dict` (`setdefault(level, []).append(i)`), sorts each bucket with `sorted(key=(kv, -i), reverse=True)`, and computes swap cost with `math.ceil` instead of integer arithmetic — a different sorting path and a different rounding path.
- 4000 random instances (`n ≤ 6`, kv ≤ 20, random bandwidth/tick parameters, about 20% protected) agree bit-for-bit on all five outputs; nine hand-computed cases (Examples 1–3, the cheap side of each mode, a tie, equal kv in one tier, stop-when-enough, infeasible, protected) plus random invariants (`freed == Σ kv[victims]`, victim priorities non-decreasing, `ticks >= replay_tokens * tick_per_token`).

## Complexity

- Bucketing plus sorting is `O(n log n)`, the main loop `O(n)`; with `n <= 64` it is constant-time in practice.
- A real engine also moves KV block page tables and issues asynchronous copy handles; only the time ledger is kept here.

## Related problems

- `inf-kv-block-admission`: decides whether a request can be admitted under the block budget; this problem decides whom to preempt when it cannot.
- `inf-lru-kv-cache-eviction`: cache-level LRU eviction, victims chosen by recency with no swap/recompute cost model.
- `inf-edf-deadline-scheduler`: deadline-ordered scheduling, a useful contrast to the priority tiers here.
- `perf-oom-boundary-search`: training-side memory-boundary search — the other half of "measure first, decide later".
