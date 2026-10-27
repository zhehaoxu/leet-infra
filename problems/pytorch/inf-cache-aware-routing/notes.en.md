# Solution Notes · Cache-Aware Routing

## Unify the units first

`alpha * missed_blocks + beta * queue_depth` mixes "blocks" with "requests". That is not a physical unit mismatch but a deliberate choice of **cost currency**: `alpha / beta` is the exchange rate between "one more request waiting" and "one more block of prefill to recompute". Tuning means moving that ratio, not the absolute magnitudes; understanding this matters more than memorising the formula.

## Why a prefix hit must stop at the first miss

A paged prefix cache can only reuse a **contiguous run starting at token 0**. Implement it as "how many of the request's blocks are in the cache" and `[10, 99, 11]` against `{10, 11}` counts as 2 hits, even though block 99 breaks the run: the third block has no reusable KV in front of it and cannot be skipped. There is exactly one correct loop:

```
k = 0
while k < len(blocks) and blocks[k] in cache: k += 1
```

## State must advance, otherwise the decision is one-shot

After routing, two things happen: `queue[chosen] += 1`, and **all** of the request's blocks go into that replica's cache (not just the matched part). Skip the write-back and the second request with the same prefix is compared as if nobody had it — stickiness disappears. Skip the queue bump and the load term never moves, making `beta` decorative.

## The two extremes are free assertions

- `alpha = 0`: cost is only the queue, i.e. least-connections balancing, so the chosen ids rotate by queue depth.
- `beta = 0`: cost is only the miss count, i.e. cache-first, so it sticks to the first replica that matched.

Both are self-checks and the entry point for understanding the parameters.

## Why ties must be pinned down

With `alpha = beta = 0` every cost is 0; without "ties to the smaller replica id" the result depends on iteration order or floating-point details. Such ties are common in practice (two replicas with empty caches and empty queues at startup), so it must be fixed by the spec.

## Complexity

M requests × R replicas × at most L blocks per row: O(M · R · L). A production router runs on a microsecond path and precomputes hit counts into hash tables, but the judging contract is the same.

## Related problems

- `inf-prefix-cache-block-hit`: block-level hit rate inside a single replica.
- `inf-continuous-batching-scheduler`: queueing and batching inside a replica.
- `inf-kv-residency-billing`: turning "blocks · steps" into money — the same resource view.
