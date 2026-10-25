# Collective Communication: All-Gather

## Problem

**All-Gather** is the most fundamental collective communication primitive: each rank holds a "local shard", and after the operation **every rank obtains the full result formed by concatenating all shards in rank order**. It is the low-level building block behind tensor-parallel forward passes and data-parallel parameter collection.

> There is no real process group in this problem. A collective is modeled as a **pure function**: the input is "each rank's local data before communication" (a list whose $r$-th item is rank $r$'s shard) and the output is "each rank's data after communication" (likewise a list whose $r$-th item is what rank $r$ receives). What you implement is this mapping.

## Rules

- Implement `solve(shards)`, where `shards` is a list of length $W$ and `shards[r]` is a one-dimensional tensor held by rank `r` (all ranks have equal length).
- After All-Gather, every rank gets the concatenation in rank order:
  $$ \text{full} = [\ \texttt{shards[0]} \Vert \texttt{shards[1]} \Vert \cdots \Vert \texttt{shards[}W\text{-1]}\ ] \quad (\dim 0 \text{ concatenation}) $$
- Return a **list of length $W$**, every element of which equals `full` (all ranks get the same result).
- Do not modify the input tensors.

## Examples

```
Input: shards = [ [10, 11], [20, 21], [30, 31] ]   # 3 ranks, each holding length 2
  full = [10, 11, 20, 21, 30, 31]
Output: [ [10,11,20,21,30,31], [10,11,20,21,30,31], [10,11,20,21,30,31] ]   # 3 identical copies
```

## Constraints

- `1 ≤ W ≤ 64`; every shard is one-dimensional, of equal length, and at least length 1
- Return `list[torch.Tensor]` of length $=W$, each shaped $(W\cdot\text{len},)$
- Shard dtype is mostly float32; equality is checked element-wise with `torch.equal`
