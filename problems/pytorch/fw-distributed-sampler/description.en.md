# DistributedSampler Index Accounting (Single-Process Simulation)

## Problem

In multi-GPU training each rank eats only \(1/\text{world}\) of the data, but each epoch's shuffle must be globally consistent and the sequence must divide evenly into the world size (\(\lceil N/R\rceil\) padded or \(\lfloor N/R\rfloor\) truncated shares) — all three ledgers (epoch determines the permutation, padding duplicates, per-rank slice offsets) are managed in one place by `DistributedSampler`. This problem runs it single-process by passing `num_replicas`/`rank` explicitly, with no process group needed, and audits its index assignment thoroughly.

## Rules

Implement `solve()` and, following the fixed setup in the starter docstring (\(N=7\), `num_replicas=3`, `seed=0`), produce five ledgers: `num_samples`/`total_size`, the replayable epoch0/epoch1 permutation slices, same-epoch replay consistency, the truncated ledger with `drop_last=True`, and the static slicing ledger with `shuffle=False`.

The sampler's semantics can be independently verified: permutation seed \(=\texttt{seed}+\texttt{epoch}\); padding \(\texttt{total\_size}=\lceil N/R\rceil\cdot R\) where the duplicated samples are the leading entries of the permutation; rank \(r\) takes \(\texttt{full}[r::R]\).

## Examples

```
Input: none
Output:
{
 "num_samples": 3, "total_size": 9,
 "epoch0_ranks": [[4, 3, 1], [0, 2, 4], [5, 6, 0]],
 "epoch1_ranks": [[0, 3, 4], [6, 5, 0], [1, 2, 6]],
 "epoch0_again": [[4, 3, 1], [0, 2, 4], [5, 6, 0]],
 "drop_last_epoch0_ranks": [[4, 3], [0, 2], [5, 6]],
 "no_shuffle_ranks": [[0, 3, 6], [1, 4, 0], [2, 5, 1]]
}
```

Deriving epoch0: the permutation is \([4,0,5,3,2,6,1]\); padding two duplicated leading entries \([4,0]\) gives the full sequence \([4,0,5,3,2,6,1,4,0]\); rank1 takes positions 1,4,7 → \([0,2,4]\).

## Constraints

- All returned lists are Python `int` lists (exactly what `list(sampler)` yields); compared exactly.
- No dependence on the global RNG: the sampler uses its own internal `torch.Generator`, and reruns must produce bitwise-identical results.
