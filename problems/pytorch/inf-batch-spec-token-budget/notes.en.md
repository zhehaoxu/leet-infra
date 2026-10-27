# Solution Notes · Per-Batch Token Budget Allocation

## Why greedy is optimal

`e_i(γ)` is **concave** in γ: its increments `α, α², α³, ...` are non-increasing for α ∈ [0,1]. Each increment costs exactly one slot, so the whole problem reduces to "pick the B largest elements from several non-increasing sequences". The optimum is "always take the current maximum" — the exchange argument closes immediately: if an optimal solution skipped the global maximum at some step, swapping it in cannot make things worse.

This is also why no knapsack DP is needed: DP only becomes necessary once the per-increment **costs differ**.

## Base allocation and incremental allocation are two steps

Hand out the N mandatory slots first (`γ_i = 0`), then distribute what is left. Looping "budget times, pick the max" from zero misses the base and over-allocates; doing the incremental loop but forgetting the `γ + 1` cost makes the total equal `budget` while the real occupancy is `budget + N`.

## The marginal exponent is `γ + 1`

Going from `γ` to `γ + 1` adds `α^(γ+1)`, not `α^γ`. That off-by-one shifts every request's allocation: the larger γ already is, the more visible it gets (with α=0.8, using 0.8 instead of 0.64 keeps piling slots onto the same request).

## Tie handling must be explicit

`gain > best_gain` with a scan from low to high index gives "ties to the smallest index" for free. Writing `>=` flips it to "ties to the largest index", which contradicts the contract and fails immediately on the all-equal-α case.

## Complexity

One scan over the batch per allocated slot: O((budget - N) × N). N heaps bring it to O(budget log N), but a batch holds only tens of requests, so the linear scan is the clearer choice.

## Related problems

- `spec-acceptance-speedup`: the same closed-form ledger for a single request, solved for the optimal γ under a latency ratio.
- `spec-sequoia-optimal-tree`: single-request tree DP that shapes the draft tree.
- `inf-kv-block-admission`: same "budget + return" flavour, with KV blocks as the resource and admission as the question.
