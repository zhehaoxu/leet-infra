# Collective Communication: All-to-All

## Problem

**All-to-All** generalizes Scatter / All-Gather: each rank holds a batch of data **already grouped by destination rank**; after communication, "the group destined for rank j" is delivered to rank j, so rank j collects **everything all ranks sent to it**. Routing tokens to the corresponding experts in MoE expert parallelism, and layout conversions in tensor parallelism, are at their core this operation.

> Collectives are modeled as pure functions: the input is a list of "each rank's local data before communication" (the `q`-th item = rank `q`), and the output is a list of "each rank's data after communication".

## Semantics

Each rank `q` holds a one-dimensional tensor `inputs[q]` (of length $W\cdot k$), split evenly into $W$ shards; the `j`-th shard `inputs[q][j*k:(j+1)*k]` is **the data rank `q` wants to send to rank `j`**. Arranging these shards into a $W\times W$ grid (rows = source rank, columns = destination rank), All-to-All is **the transpose of this grid**:

$$ \text{out}[j] \;=\; \bigoplus_{q=0}^{W-1} \texttt{inputs}[q][\,j\cdot k:(j+1)\cdot k\,] $$

($\oplus$ denotes concatenation in source-rank order `q`). That is, rank `j` receives the **`j`-th column** of the grid.

## Rules

Implement `solve(inputs)`: `inputs` is a list of length $W$, each item a one-dimensional tensor whose length is divisible by $W$. Return a **list of length $W$**, where the `j`-th item = the shards all ranks sent to `j`, concatenated in source-rank order. Do not modify the inputs.

## Examples

```
W=2, k=2:
Input: inputs = [ [10,20 | 30,40],                 # rank0: →0:[10,20], →1:[30,40]
                [50,60 | 70,80] ]                 # rank1: →0:[50,60], →1:[70,80]
Output: [ [10,20,50,60],                             # rank0 collects all "sent to 0" (from rank0, rank1)
        [30,40,70,80] ]                            # rank1 collects all "sent to 1"
```

## Constraints

- `1 ≤ W ≤ 64`; each item has length $=W\cdot k$ (`k ≥ 1`), and all ranks use the same shard size
- Return `list[torch.Tensor]` of length $=W$, each shaped $(W\cdot k,)$
- Special cases: if all ranks fill only their diagonal shard → degenerates to Scatter; if each rank's $W$ shards are all identical → degenerates to All-Gather
