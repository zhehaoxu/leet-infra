# Distributed: Elastic Fault-Tolerance Rank Remapping and Shard Migration

## Description

After some ranks fail, renumber the survivors and restore the failed ranks' original parameter shards from a checkpoint. The original `world_size` ranks each own one equal shard, with `param_bytes` total parameter bytes.

**Recovery assumes a consistent durable checkpoint containing every failed shard.** Survivors retain their own shards at that same checkpoint state; failed shards are read from the checkpoint. If unique data existed only in lost RAM with no durable copy, remapping cannot recover it.

## Remapping and restoration rules

1. Deduplicate failed IDs. Input order does not affect the result, and duplicate IDs do not trigger duplicate restoration.
2. Retain survivors in **ascending old-ID order**, assigning new IDs \(0,1,\ldots,W'-1\), where \(W'\) is the number of survivors. Failed entries in the old-to-new map are -1.
3. Process checkpoint shards in **sorted unique failed-ID order**. Starting at new rank 0, assign the \(i\)-th failed shard to new rank \(i\bmod W'\), with zero-based indexing. Destinations are new IDs, not old IDs.
4. Do not move the survivors' existing shards. `migration_bytes` counts **only the failed-shard checkpoint restore payload**.

For the failed-ID set \(F\) and original shard size \(s=\text{param_bytes}/\text{world_size}\),

\[
W'=\text{world_size}-|F|,
\qquad \text{migration bytes}=|F|s.
\]

Original shards must be equal, so total bytes must divide evenly by the original rank count. No failures require no restoration. If every rank fails, there is no recipient and this model cannot proceed. The starter specifies exact types, ranges, and `ValueError` behavior.

This is a deterministic checkpoint-restore ledger, **not torchelastic's real rank reassignment algorithm**. It does not simulate a full restart or repartition every shard for the new world size.

## Examples

```text
world_size=4, failed_ranks=[1, 3], param_bytes=16
Surviving old ranks [0, 2] -> new ranks [0, 1]
Failed checkpoint shard 1 -> new rank 0; shard 3 -> new rank 1
Each shard is 4 bytes; only two failed shards are restored, totaling 8 bytes
Return (2, [0, -1, 1, -1], [0, 1], 8)
```

```text
world_size=8, failed_ranks=[7, 2, 5, 0, 4], param_bytes=64
Surviving old ranks [1, 3, 6] -> new ranks [0, 1, 2]
Sorted failed shards [0, 2, 4, 5, 7] have destinations [0, 1, 2, 0, 1]
Return (3, [-1, 0, -1, 1, -1, -1, 2, -1], [0, 1, 2, 0, 1], 40)
```

The second example gives survivors `[2, 2, 1]` extra shards. Only failed shards contribute restore payload; the three existing survivor shards are excluded from migration bytes.
