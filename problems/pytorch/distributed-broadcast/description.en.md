# Collective Communication: Broadcast

## Problem

**Broadcast** lets a designated source rank (`src`) copy its tensor verbatim to **all** ranks. It is the simplest collective: synchronizing initial weights at the start of training and distributing configuration during evaluation both rely on it.

> Collectives are modeled as pure functions: the input is a list of "each rank's local data before communication" (the $r$-th item is rank $r$'s tensor; only the `src` copy is valid data and the rest are placeholders), and the output is a list of "each rank's data after communication".

## Rules

- Implement `solve(inputs, src)`: `inputs` is a list of length $W$; `inputs[src]` is the source rank's valid tensor (of arbitrary shape), and the tensor contents of the other ranks **are not used** (placeholders).
- Result: after communication **every rank holds a copy of `inputs[src]`**.
- Return a **list of length $W$**, where the `r`-th item equals `inputs[src]`.
- Do not modify the input tensors.

## Examples

```
Input: inputs = [ [99], [99], [1.0, 2.0, 3.0] ], src = 2   # rank2 is the source
Output: [ [1,2,3], [1,2,3], [1,2,3] ]                        # every rank copies the source data
```

## Constraints

- `1 ≤ W ≤ 64`; `0 ≤ src < W`
- `inputs[src]` may have any shape; the other items may have different shapes (they are not used anyway)
- Return `list[torch.Tensor]` of length $=W$, each item `torch.equal` to `inputs[src]`
- Difference from All-Gather: Broadcast merely **copies the same copy**, without concatenating different shards
