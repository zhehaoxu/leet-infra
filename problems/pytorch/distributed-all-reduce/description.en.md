# Collective Communication: All-Reduce

## Problem

**All-Reduce** = element-wise reduction (summation) over the data of all ranks, with the result **appearing on every rank simultaneously**. It is the core primitive for synchronizing gradients at every step of data parallelism (DDP), and it equals "Reduce-Scatter followed by All-Gather".

> Collectives are modeled as pure functions: the input is a list of "each rank's local data before communication", and the output is a list of "each rank's data after communication".

## Rules

- Implement `solve(inputs)`, where `inputs` is a list of length $W$ and `inputs[r]` is rank `r`'s tensor; **all tensors have the same shape** (one- or multi-dimensional).
- Sum element-wise to obtain the global result:
  $$ T = \sum_{r=0}^{W-1} \texttt{inputs[}r\texttt{]} $$
- Return a **list of length $W$**, every element of which equals $T$ (the reduction result is broadcast to all ranks — this is the key difference from Reduce-Scatter: here everyone gets the **complete** result).
- Do not modify the input tensors.

## Examples

```
Input: inputs = [ [[1,2],[3,4]], [[10,20],[30,40]] ]   # W=2, shape (2,2)
  T = [[1+10, 2+20], [3+30, 4+40]] = [[11,22],[33,44]]
Output: [ [[11,22],[33,44]], [[11,22],[33,44]] ]        # both ranks identical
```

## Constraints

- `1 ≤ W ≤ 64`; all input tensors have identical shapes
- Return `list[torch.Tensor]` of length $=W$, each with the same shape as the input
- If data parallelism needs **gradient averaging**, that is post-processing that divides by $W$ uniformly after All-Reduce (summation); this problem only sums
- Equality is checked with `torch.allclose` (floating point) / `torch.equal` (integer)
