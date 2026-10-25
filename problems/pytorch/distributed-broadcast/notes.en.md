# Approach

## Interviewer's Intent

Broadcast is the simplest yet ubiquitous collective: at DDP initialization, rank0's model parameters are broadcast to all GPUs; `torch.distributed.broadcast` synchronizes the loss scaler; a scalar configuration is distributed. Only by understanding it can you distinguish it from All-Gather — both "let all ranks obtain something", but the data sources are completely different.

## Semantics

Given a designated source rank `src`, only `inputs[src]` is valid data and the other ranks' inputs are placeholders. After communication **every rank equals a copy of `inputs[src]`**. Note that it is a **one-to-many copy** and performs no merging or concatenation.

## Comparison with All-Gather / Scatter

- **Broadcast**: one copy of data → all ranks (the same copy, repeated).
- **All-Gather**: each rank has a different shard → all ranks receive "the concatenation of everything".
- **Scatter** (see `distributed-scatter`): src holds the complete data → split into $W$ shards distributed to the ranks (the "sliced version" of Broadcast).

## Implementation Notes

Simply take `inputs[src]` and copy it $W$ times. Use `.clone()` to ensure no storage is shared between ranks or with the input, thereby satisfying "do not modify the input". `src` need not be 0; do not hard-code `inputs[0]` — this is what `check_src_index_used` tests.

## Real Communication Volume

To broadcast $B$ bytes of data, the naive approach has src send $W-1$ times for a total of $(W-1)B$; NCCL uses tree broadcast to reduce **latency** to $O(\log W)$ (parallel forwarding at each level), achieving higher bandwidth utilization. This is why distributing weights in large clusters does not become a bottleneck.

## Grading Approach

`test_cases.py` verifies: correct use of the `src` index (including src≠0), 2-D data with placeholder lengths that differ, single-rank degeneration, and no modification of the inputs.
