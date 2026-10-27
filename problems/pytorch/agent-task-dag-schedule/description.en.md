# Agent Infrastructure: Task DAG Topological Execution and Critical Path

## Problem

Multi-agent frameworks (HuggingGPT, BabyAGI, MetaGPT) split a request into a dependency graph of sub-tasks and hand it to a scheduler: tasks without dependencies can run in parallel, tasks with dependencies must wait for their predecessors. The scheduler answers three questions —

- **Execution order**: what is a legal topological order (the order actually dispatched to executors);
- **Parallelism**: how many tasks can run at once (the wave partition), which sizes the worker pool;
- **Makespan lower bound**: even with unlimited parallelism, the critical path bounds the total time.

None of the three falls out for free: the topological order needs an explicit tie-break, the waves must follow the longest predecessor chain, and the makespan must be weighted by task duration.

## Task

Implement `solve(n, edges, cost)` returning `(order, waves, crit)`:

- Tasks are `0..n-1`; `(u, v)` in `edges` means `u` must precede `v`; `cost[i]` is the duration of task `i` (`> 0`).
- `order`: Kahn's algorithm, always taking the **smallest index** in the ready set so the result is unique.
- `waves`: `level[v] = 0` for sources, otherwise `1 + max(level[u] for u -> v)`; wave `k` holds every task with `level == k`, sorted ascending inside the wave.
- `crit`: the maximum `cost` sum along any dependency chain (`float`).
- A cycle (including a self-loop), an out-of-range index, a `cost` of the wrong length or non-positive → `ValueError`.

## Example

```
n, cost = 6, [3, 2, 4, 1, 5, 2]
edges = [(0, 2), (1, 2), (2, 3), (2, 4), (4, 5)]

solve(n, edges, cost) -> (
    [0, 1, 2, 3, 4, 5],
    [[0, 1], [2], [3, 4], [5]],
    14.0,
)

critical chain 0 -> 2 -> 4 -> 5: 3 + 4 + 5 + 2 = 14
```

## Constraints

- **The order needs a fixed tie-break**: when several tasks are ready, take the smallest index — otherwise the same graph yields different orders in different implementations and deterministic judging is impossible.
- **Waves are not a chunking of the topological order**: tasks within a wave are mutually independent, and every predecessor must sit in an earlier wave (`level` takes the longest predecessor chain, not the shortest).
- **The critical path is a weighted longest path**: the chain with the most hops is not necessarily the slowest; in a diamond the heavier branch must win, and an isolated task is itself a chain of its own cost.
- **Duplicate edges change nothing**: `(0, 1)` twice equals once, so either deduplicate or count it twice in both the indegree and the out-edge walk.
- **Cycles must be detected first**: after Kahn, an output shorter than `n` means a cycle; a self-loop is a cycle too and should fail while building the graph.
