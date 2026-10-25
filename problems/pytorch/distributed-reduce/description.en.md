# Collective Communication: Reduce (Root Reduction)

## Problem

**Reduce** is the most basic collective communication primitive: it reduces all ranks' data element-wise (summation here), and the result **is collected only on the designated root rank (root)**; the buffers of the other ranks remain unchanged. All-Reduce is exactly "Reduce + Broadcast", and Reduce-Scatter is "Reduce, after which each rank keeps only one segment". Understanding Reduce's "only the root gets the complete result" is the starting point of this family of primitives.

> Collectives are modeled as pure functions: the input is a list of "each rank's local data before communication" plus the root rank index, and the output is a list of "each rank's data after communication".

## Rules

- Implement `solve(inputs, root)`, where `inputs` is a list of length $W$ and `inputs[r]` is rank `r`'s tensor (**all tensors have the same shape**, one- or multi-dimensional); `root` ($0\le root<W$) is the destination rank for the aggregation.
- Sum element-wise to obtain the global result:
  $$ T = \sum_{r=0}^{W-1} \texttt{inputs[}r\texttt{]} $$
- Return a **list of length $W$**: the `root`-th item equals $T$; **every other item keeps that rank's original input unchanged** (the key difference from All-Reduce — the result is not broadcast to everyone, only placed on the root).
- Do not modify the input tensors.

## Examples

```
Input: inputs = [ [[1,2],[3,4]], [[10,20],[30,40]] ],  root = 0   # W=2, shape (2,2)
  T = [[1+10, 2+20], [3+30, 4+40]] = [[11,22],[33,44]]
Output: [ [[11,22],[33,44]],   # root=0 gets the complete sum
        [[10,20],[30,40]] ]  # rank 1 keeps its own original values
```

## Constraints

- `1 ≤ W ≤ 64`; `0 ≤ root < W`; all input tensors have identical shapes
- Return `list[torch.Tensor]` of length $=W$, each with the same shape as the input
- Only summation reduction (`SUM`) is implemented; non-root ranks keep their original values, are not zeroed, and receive no broadcast
- Equality is checked with `torch.allclose` (floating point) / `torch.equal` (integer)
