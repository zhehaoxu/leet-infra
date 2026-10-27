# Solution Notes · Task DAG Scheduling

## Three outputs, three separate problems

- `order` is the **execution protocol**: the sequence actually dispatched to workers, which must be legal and unique;
- `waves` is **capacity planning**: how many run at once, which sizes the pool;
- `crit` is the **performance lower bound**: no amount of extra workers goes below it.

They live in one problem because all three read the same graph yet each has its own trap: computing only the order misses the critical path, computing only the critical path misses the waves.

## Why the order needs a fixed tie-break

Kahn's ready set is usually a queue, so the dequeue order depends on traversal and a graph has dozens of legal orders. Elementwise judging forces a deterministic rule: **take the smallest-index ready task**. That is equivalent to a min-heap, and also to "scan once, pick the first task whose predecessors are all done" — slower but easier to verify, which makes it a perfect oracle.

A related bug: treating "indegree zero" as "runnable now" and dumping all sources at the front, which hides the fact that they could have run in parallel.

## Waves = longest predecessor chain

`level[v] = 1 + max(level[u])` takes the **longest**, not the shortest: if A has a two-deep predecessor chain, it cannot run before wave 2. Using the shortest chain silently drops the "all predecessors must finish" constraint — the single easiest line to miss in review.

The partition also yields maximum parallelism `max(len(wave))`. Note this is only the graph's own width; real concurrency is further capped by workers and memory.

## Why the critical path must be weighted

Unweighted, "longest chain" is just a hop count, equal to `max(level) + 1`, which duplicates the wave count and makes the problem degenerate. With `cost`, a diamond `0 -> 1 -> 3` (5+1) versus `0 -> 2 -> 3` (2+1) must pick the former — exactly where "short tasks go wide, the long task owns the critical path" comes from in distributed jobs.

In practice both the path and the levels come out of one pass: on dequeue, push `best[u] + cost[u]` to each successor taking the max, then take `max(best[v] + cost[v])`.

## Complexity and real systems

O(V + E log V) with a heap, or O(V²) with the naive scan. Production schedulers additionally handle dynamically appended tasks, unknown runtimes (estimated from historical quantiles), retries with idempotency, and cross-machine transfer costs — all built on the same skeleton of order + waves + critical path.

## Related problems

- `agent-react-loop`: the single-action loop; this problem is its multi-task version.
- `agent-context-compaction`: more sub-tasks mean a tighter context.
- The `distributed` collection: the cross-device version of waves and critical paths.
