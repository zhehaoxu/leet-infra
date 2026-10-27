# Cache-Aware Sticky Routing: Prefix Hits vs Queue Depth

## Problem

A router in front of several replicas has to please two goals at once:

- **Hit the prefix cache**: a given prefix was precomputed on exactly one replica; sending the request elsewhere forces a full prefill recompute. The longer the hit, the more TTFT you save.
- **Do not pile onto one replica**: if every request chases the replica with the fullest cache, its queue grows without bound and tail latency collapses.

Production systems (Mooncake, the SGLang router, vLLM production-stack) express both as one **weighted cost**: `α · missed_blocks + β · queue_depth`, then pick the cheapest replica. This problem is the deterministic version of that cost, with ties broken by the smaller replica id, and with cache and queue state carried forward after every decision.

## Task

Implement `solve(cache, queries, queue, alpha, beta)`:

- Inputs:
  - `cache` `(R, C)` int64, row r is the prefix block hashes replica r already holds (`-1` is padding only);
  - `queries` `(M, L)` int64, row j is request j's prefix block hashes (a `-1` ends the sequence);
  - `queue` `(R,)` int64 current queue depth per replica; `alpha`, `beta` are the cost weights (both `>= 0`).
- Process requests in row order. For request j and replica r:
  - `matched` = how many leading blocks of the request are **contiguously** present in replica r's cache (stop at the first miss);
  - `cost = alpha * (len_j - matched) + beta * queue[r]`.
- Pick the smallest cost, ties to the smaller replica id; then `queue[chosen] += 1` and add **all** of the request's prefix blocks to that replica's cache.
- Outputs: `chosen` `(M,)`, `hits` `(M,)`, `final_queue` `(R,)`, `total_cost` float32 scalar (sum of the chosen costs).

## Example

```
cache   = [[10, 11, 12], [10, 20, 21]]     replica 0 holds {10,11,12}, replica 1 holds {10,20,21}
queries = [[10, 11, 5],  [10, 20, 21]]
queue   = [0, 0],  alpha = 1.0,  beta = 1.0

q0 = [10,11,5]: r0 matched 2 -> cost = 1*(3-2) + 1*0 = 1
                r1 matched 1 -> cost = 1*(3-1) + 1*0 = 2   -> pick r0, queue = [1,0]
q1 = [10,20,21]: r0 matched 1 -> cost = 2 + 1 = 3
                 r1 matched 3 -> cost = 0 + 0 = 0          -> pick r1, queue = [1,1]

chosen      = [0, 1]
hits        = [2, 3]
final_queue = [1, 1]
total_cost  = 1.0
```

## Constraints

- **A hit must be a contiguous prefix**: `[10, 99, 11]` matches only 1 block against `{10, 11}`; the trailing 11 does not count — a paged prefix cache can only reuse the run starting at block 0.
- **The miss term counts blocks, not a ratio**: `len_j - matched` is exactly the number of blocks to recompute; a ratio flattens the weight of long requests.
- **Write the cache back after routing**: the chosen replica immediately holds all of the request's prefix blocks, so the next request with the same prefix hits it — that is where "sticky" comes from.
- **The queue only grows**: completion is not modelled; depth means "requests assigned so far" and exists to express load.
- **Ties go to the smaller replica id**, keeping the result implementation-independent.
- **`alpha = 0` degenerates to pure load balancing** (rotation by queue depth) and **`beta = 0` to pure cache affinity**; both extremes must work.
