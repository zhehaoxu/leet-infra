# Solution Notes · TTFT / TPOT Timeline

## Origin

- Orca: *A Distributed Serving System for Transformer-Based Generative Models* (OSDI '22) — introduces iteration-level scheduling (continuous batching): the batch is rebuilt every step and finished requests release their slots immediately.
- Sarathi-Serve: *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (arXiv:2403.02310) — chunked prefill splits long prompts and packs them with decode into the same step, trading TTFT against TPOT via the chunk size.
- vLLM and other engines expose TTFT / TPOT as standard SLO metrics; the "one action per step" model here is the minimal shared-forward abstraction.

## Key points and pitfalls

**Prefill and decode share one step.** This is the one physical constraint the problem cannot drop: on a prefill step no running request emits a token. In Example 2 the four prefill steps of r1 stretch r0's three token gaps to seven steps — precisely why chunked prefill exists. Letting prefill and decode run in parallel would make TTFT look good while the TPOT ledger is simply wrong.

**Admission happens at the top of a step; a released slot is usable only next step.** A request finishing in step t's decode releases its slot at the end of step t; the admission phase of step t+1 is the first that can use it. Get this order wrong and every request's first token arrives one step early.

**Admission order is `(arrival, index)`, not array order.** Arrivals may be out of order, so sort them first; ties on `arrival` are broken by index to keep the schedule deterministic.

**Prefill is FCFS and advances one request at a time.** Among active requests the target is the earliest admitted one that still has prompt tokens; you may neither pack several prefills into one step nor let a newer request jump the queue.

**The step that finishes a prefill emits no token.** The first token comes in the following decode step; a request with `gen_len = 1` emits first and last token in the same step, so `decode_span = 0`.

**Idle time must jump the clock.** When the active set is empty you cannot spin step by step (`arrival` can be large); jump straight to the next arrival, and idle steps count toward neither `prefill_steps` nor `decode_steps`.

**The two counters are invariants.** `prefill_steps` always equals `Σ ceil(prompt_len_i / chunk)` (every request's prefill is eventually completed); `decode_steps` lies between `max(gen_len)` and `Σ gen_len`.

## Verification

- Reference: a single `while` loop with an admission list and a dict of remaining prefill, branching into prefill / decode / idle and `continue`ing.
- Independent oracle: a `deque` prefill queue plus a `list` decode set — advance the queue head, migrate finished prefill into the decode set; data structures and control flow differ from the reference.
- On 3000 random `(arrivals, prompt_lens, gen_lens, chunk, max_batch)` cases the two agree exactly on all five outputs; layered on top are hand-computed timelines (Examples 1–3) and invariants (`ttft >= ceil(prompt/chunk)`, `finish = arrival + ttft + span`, the `prefill_steps` identity).

## Complexity

- `O(active requests)` per step, with total steps bounded by `Σ ceil(prompt_len_i / chunk) + Σ gen_len_i` — `O((Σ prefill chunks + Σ gen_len) × n)` overall; at `n <= 64` this is millisecond-scale.

## Related problems

- `inf-continuous-batching-scheduler`: computes finish steps only, with no prefill time — the degenerate version of this problem with the prefill ledger removed.
- `inf-chunked-prefill-scheduling`: chunks the prompt by a token budget without any timeline.
- `inf-batch-spec-token-budget`: allocates a budget inside one step; complementary to this problem's between-step timeline.
