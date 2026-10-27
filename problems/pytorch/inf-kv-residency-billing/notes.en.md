# Solution Notes · KV Residency Integral and Billing

## The integral is not a multiplication

The beginner formula is `residency = out_tokens * ceil((prompt_len + out_tokens) / block_size)`. That multiplies counts using the **final** (fullest) occupancy, so it is systematically too high. The right picture is a step-by-step Riemann sum: at step k the occupancy is `ceil((prompt_len + k) / block_size)`, added one term at a time.

## Closed form: build a prefix sum of the staircase

Looping is fine when `out_tokens` is modest, but for O(1) construct

```
G(T) = Σ_{t=0..T} ceil(t / bs) = bs*K*(K+1)/2 + R*(K+1),  K = T // bs, R = T % bs
```

then `residency(p, m) = G(p + m - 1) - G(p - 1)`. The intuition: `ceil(t/bs)` partitions the t axis into steps of length `bs`; each full step contributes "level × length", and the partial last step is settled by its remaining length.

Note `G(-1) = 0`, which is exactly why an empty interval (`out_tokens = 0`) has integral 0.

## The peak must be accumulated moment by moment

The peak is not "the max of each request's peak" but the max over time of the **sum of all active requests**. You can sweep the timeline with a dict, or use event differences (enter +size, leave -size) — except the block count itself changes, so differences are not enough and the robust route is adding each request's per-step occupancy into `per_step[t]`.

## Two consequences of the half-open interval

- Blocks are released right after the last token, so the number of occupied steps equals `out_tokens`, with no extra one.
- Iterate `t = start + k, k ∈ [0, out_tokens)`; writing `[start, start + out_tokens]` inflates both the integral and the peak by one step.

## Why the two charge terms cannot really be added

`price_out` is per token, `price_block_step` is per block per step. They add only because the business defines an exchange rate; the implementation must compute each term from its own quantity and then sum, never mix `out_tokens` and `residency` in one dimension.

## Complexity

The step-wise version is O(Σ out_tokens) (peak included); the closed-form integral is O(N), while the peak still needs O(Σ out_tokens). Real billing aggregates into time buckets, which is the same thing with a histogram instead of `per_step`.

## Related problems

- `inf-cache-aware-routing`: blocks as the currency of a routing decision.
- `kv-cache-memory` in `perf-systems`: static peak occupancy with no time integral.
- `inf-paged-kv-block-alloc`: allocation order, which decides how far the peak can be pushed down.
