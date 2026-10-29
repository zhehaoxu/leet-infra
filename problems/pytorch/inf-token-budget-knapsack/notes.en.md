# Solution Notes · Per-Step Token Budget Knapsack

## Origin

- Sarathi-Serve: *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (arXiv:2403.02310) — stall-free batching plus chunked prefill: decode reserves its slots, the remaining budget goes to prefill chunks, so a long prompt cannot starve running requests.
- Orca (OSDI '22) iteration-level scheduling and vLLM continuous batching are the ancestors of this budget model; production engines (vLLM, DistServe, …) layer priority / SLO weights on top, which is this problem's `values`.
- Formulating "which requests get admitted this step" as a 0/1 knapsack is the standard abstraction in this line of scheduling papers; real systems approximate it, while this problem demands the exact optimum.

## Key points and pitfalls

**Subtract decode first, then talk about prefill.** `remaining = budget - decode_count`; decode is the base load and may not be sacrificed for prefill (that is the stall-free premise). Forget the subtraction and every candidate set will swallow one or two extra requests.

**The objective is a three-key lexicographic order, not a single number.** Largest weight → smallest cost → lexicographically smallest index list. The second key matters: budget saved this round becomes next round's chunk budget, so in Example 2 `{0,2}` beats `{0,1,2}` despite equal weight.

**Zero-weight requests are never admitted.** They consume budget without adding value, and the third key (lexicographic index list) makes the empty set win. They can still receive the leftover chunk, so do not drop them from the chunk candidates.

**The leftover goes to the FCFS head, not to the cheapest request.** The admitted set comes from the knapsack, but the chunk follows queue semantics: the smallest-index unadmitted request moves first. In Example 3 index 0 is admitted while the chunk lands on index 1.

**Exactly one chunk, no cascade.** Whatever remains after the chunk is `unused`; it is not fed to the next request, mirroring an engine that advances a single prefill chunk per step.

**`n` is small, but do not degenerate to enumeration.** The constraints allow `n <= 12`, so `2^n` passes; the standard solution is a capacity DP in `O(n · remaining)` that also scales to longer queues.

## Verification

- Reference: a capacity-indexed 0/1 knapsack DP keeping `(value, ascending index tuple)` per exact cost, compared by `(-value, index tuple)` at equal cost; the final pick scans all `capacity <= remaining` states with key `(-value, capacity, index tuple)`.
- Independent oracle: enumerate all `2^n` subsets and take the best by `(-value sum, cost, index tuple)`.
- On 4000 random `(prompt_lens, values, budget, decode_count)` cases the five-tuples agree exactly, including zero weights, empty queues and infeasible instances. Additional per-output invariants: the admitted set stays within budget, the chunk never lands on an admitted request, the chunk index is the smallest unadmitted index, and `chunk_tokens + unused` conserves the leftover.

## Complexity

- DP: `O(n · remaining)` time and `O(remaining)` space; with `n <= 12` and `remaining <= 128` it is instant, and the `2^n` brute force is acceptable too.

## Related problems

- `inf-chunked-prefill-scheduling`: splits chunks by budget without deciding *which* requests to admit.
- `inf-batch-spec-token-budget`: allocates a budget within one step, but with a concave payoff where greedy is already optimal — no knapsack required.
- `inf-ttft-tpot-queue`: chains per-step decisions into a timeline to see how prefill crowds out TTFT / TPOT.
