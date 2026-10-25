# Collective Communication: Scatter

## Problem

**Scatter** is the inverse of All-Gather: the source rank holds one **complete copy of data**, and after communication it is **split evenly into $W$ segments**, with the `i`-th segment sent only to rank `i`. Every rank is thus left with only its own segment. It is commonly used to slice a full batch and distribute it across GPUs.

> Collectives are modeled as pure functions: the input is the source rank's complete tensor and the number of ranks, and the output is a list of "each rank's data after distribution".

## Rules

- Implement `solve(full, world)`: `full` is a one-dimensional tensor (length $L$ divisible by `world` $=W$), and `world` is the number of ranks.
- Split `full` evenly into $W$ segments, each of size `chunk = L / W`; rank `i` receives the `i`-th segment:
  $$ \text{out}_i = \texttt{full}[\,i\cdot\text{chunk} : (i+1)\cdot\text{chunk}\,] $$
- Return a **list of length $W$**, where the `i`-th item is the segment rank `i` receives (ranks get different results).
- Do not modify the input tensor.

## Examples

```
Input: full = [1,2,3,4,5,6], world = 3     # L=6, chunk=2
Output: [ [1,2], [3,4], [5,6] ]             # rank0←first segment, rank1←middle, rank2←last
```

## Constraints

- `1 ≤ world ≤ 64`; `full` is one-dimensional and its length is divisible by `world` (chunk ≥ 1)
- Return `list[torch.Tensor]` of length $=W$, each shaped $(L/W,)$
- Difference from Reduce-Scatter: Scatter **performs no cross-rank summation**; it only slices and distributes one copy of data
