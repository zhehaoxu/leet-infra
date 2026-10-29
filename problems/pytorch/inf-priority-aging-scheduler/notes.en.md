# Solution Notes · Priority Admission Scheduling with Aging

## Origin

- Priority scheduling and starvation are textbook OS material (the priority-scheduling chapter of Silberschatz, *Operating System Concepts*): static priorities let low-priority tasks wait forever, and **aging** — turning waiting time into a priority increment — is the textbook remedy.
- On the inference side, vLLM exposes per-request `priority` with priority-ordered scheduling, and continuous batching (Orca, OSDI '22) makes "re-batch every step" the norm; the admission loop here is an abstraction of that layer.
- `priority_cap` mirrors the "cap the aging" knob used in practice: without a cap, long-waiting low-priority requests overtake everything else and the SLO tiers collapse.

## Key points and pitfalls

**Recompute the effective priority at every pick.** A single step may hand out several slots; freezing the value once at the top of the step would serve later slots with stale numbers. All picks here share the same `t`, but the formula must be evaluated with `t` rather than a cached result.

**Aging is integer division.** `(t − arrivals[i]) // aging_interval`: waiting less than one interval adds nothing (in Example 1 at step 1, r1's effective priority is still 1, not 1.0 or 2).

**Ties have three levels.** Effective priority → base priority → arrival time → index. Step 5 of Example 1 uses the second level (base 5 > 0) so r6 beats r1, which had already aged five levels; step 0 of Example 3 uses levels three and four.

**The cap eats aging.** In Example 2 `priority_cap = 5`, so the low-priority request can at best tie at 5, and the tie-breaker always favours the higher base priority — it only gets in when the stream ends. Aging without headroom is no aging.

**Slots are released at the end of the step.** A request finishing at step `t` frees its slot for step `t+1` (Example 3). Get this wrong and every request is admitted one step early.

**Idle means jumping the clock.** The first request may arrive at a large step number; spinning one step at a time is not acceptable.

**`finish` is an invariant.** `finish[i] = admit[i] + service[i] − 1` always holds; it also verifies that service occupies consecutive, uninterruptible steps.

## Verification

- Reference: a `waiting` set plus a `running` dict (remaining service steps); each step tops up the slots then advances all running requests, and idle time jumps with `min(arrivals)`.
- Independent oracle: an arrival-sorted `pending` list, a `running` list and a `left` array; each pick uses `max()` over `(eff, base, −arrival, −index, i)` tuples — an `O(n²)` naive scan with no sorting and no sets.
- 4000 random instances (`n ≤ 7`, arrivals 0–8, service 1–4, priority 0–5, batch 1–3, aging interval 1–4, cap randomly above the maximum priority) agree bit-for-bit; six hand-computed scenarios (Examples 1–3, no headroom, aging-interval contrast, tie-breaking, multiple slots, idle jump) plus random invariants (`admit >= arrival`, `finish == admit + service − 1`, per-step concurrency ≤ `max_batch`, `max_wait == max(admit − arrival)`).

## Complexity

- One `O(n)` scan per slot; the number of steps is bounded by `Σ service[i] + max(arrivals)`, so the total is `O(n × steps)` — milliseconds for `n ≤ 64`.
- A production scheduler uses heaps or ordered trees to make each pick `O(log n)` and reorders lazily by aging level.

## Related problems

- `inf-continuous-batching-scheduler`: plain FCFS continuous batching with neither priority nor aging.
- `inf-edf-deadline-scheduler`: earliest-deadline-first, an alternative anti-starvation idea based on slack rather than waiting time.
- `inf-preempt-swap-recompute`: the preemption call when admission fails — complementary to the queue ordering here.
- `inf-ttft-tpot-queue`: the prefill/decode timeline after admission; this problem only decides who runs first.
