# Collective Communication: Gather

## Problem

**Gather** is the "single-destination version" of All-Gather: every rank hands over its own shard, but the concatenated **complete result is delivered only to one destination rank** (`dst`); the other ranks do not receive the whole and **keep their original shard unchanged**. A real use case is each GPU gathering local statistics to rank0 for aggregation without broadcasting back to everyone.

> Collectives are modeled as pure functions: the input is a list of "each rank's local data before communication" (the `q`-th item = rank `q`), and the output is a list of "each rank's data after communication".

## Rules

Implement `solve(inputs, dst)`: `inputs` is a list of length $W$, each item a one-dimensional tensor of **equal length**; `dst` is the destination rank. Return a **list of length $W$**, where:

- the `dst`-th item $= \bigoplus_{q=0}^{W-1} \texttt{inputs}[q]$ (the shards of all ranks concatenated in rank order into the complete result);
- the `j`-th item (`j != dst`) $= \texttt{inputs}[j]$ (**keeps its own shard as-is**; gather does not give them the result).

Do not modify the input tensors.

## Examples

```
Input: inputs = [ [1,2], [3,4], [5,6] ], dst = 1
Output: [ [1,2],                      # rank0 non-destination: keeps its shard
        [1,2,3,4,5,6],              # rank1=dst: collects the full concatenation
        [5,6] ]                     # rank2 non-destination: keeps its shard
```

## Constraints

- `1 ≤ W ≤ 64`; `0 ≤ dst < W`; all shards have the same length
- Return `list[torch.Tensor]` of length $=W$; `out[dst]` has length $=W\cdot\text{shard}$, and every other item equals the corresponding input
- Difference from **All-Gather**: All-Gather gives the complete result to **all** ranks, whereas Gather gives it to **only the `dst`** rank
- Difference from **Reduce**: Gather is **concatenation** (no aggregation); numerically it does no summation
