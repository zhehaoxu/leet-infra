# In Depth: Reduce (Root Reduction)

## Where Reduce Sits in the Collective Family

Using "who ultimately gets the result" as the classification axis, the reduction-type primitives become clear at a glance:

| Primitive | Result destination | Result per rank |
|---|---|---|
| **Reduce** | only the root | root gets the complete sum, the rest are **unchanged** |
| **All-Reduce** | all ranks | everyone gets the complete sum |
| **Reduce-Scatter** | all ranks, but each keeps one segment | each gets a local sum of size $1/W$ |
| Scatter/Gather | one-to-many / many-to-one | distribute / collect |

Identities: `All-Reduce = Reduce + Broadcast`; `All-Reduce = Reduce-Scatter + All-Gather`. This problem does the most primitive step — **reduce to root**.

## Modeling Notes

- Pure-function view: the arguments are the list of each rank's local tensors before communication plus the root index, and the return value is the list after communication.
- **Non-root ranks keep their original input** (not zero, not the sum). This is the semantic core that distinguishes Reduce from All-Reduce, and the most common pitfall (writing it as "broadcast to everyone") — `check_only_root_has_sum` checks item by item that non-root entries still equal their inputs.
- The root entry is the **complete** element-wise sum (of the same shape as the input), not some shard.

## Implementation Pitfalls

- **Do not modify inputs in place**: before `out[root] = total`, first copy the list into `[t.clone() for t in inputs]`, otherwise the input is polluted (`check_input_not_mutated` catches this).
- The summation path is free (`stack().sum()` or sequential accumulation); floating point is compared with `allclose`, integers can be compared exactly with `torch.equal`.
- When `W=1` the root is the only rank and the result is itself — the degenerate case must hold.

## Related

- Upstream/contrast: `distributed-all-reduce` (one more broadcast step), `distributed-reduce-scatter` (result sharded).
- Downstream: `distributed-collective-latency` analyzes the elapsed time of such reductions under ring / recursive-doubling.
- Grading: pure CPU tensor simulation + an independent recomputation in `test_cases`, with no GPU/multi-process dependency.
