# One-Hot Encoding

## Problem

Encode integer class indices as one-hot vectors: **append a new class dimension** at the end, with the indexed position set to 1 and the rest to 0.

$$ \text{onehot}(v)_c = \begin{cases} 1 & c = v \\ 0 & \text{otherwise} \end{cases} $$

## Rules

- Implement `solve(indices, num_classes=None)`; the output shape is `(*indices.shape, num_classes)` with dtype `int64` (`torch.long`).
- When `num_classes=None`, use $\max(\text{indices})+1$; when given explicitly, use the given value (extra columns are filled with 0; if it is smaller than $\max+1$, indexing is out of bounds and raises an error).
- Semantics match `F.one_hot`.

## Examples

```
Input: indices=[1,3,2]  → num_classes=max+1=4
Output: [[0,1,0,0],
      [0,0,0,1],
      [0,0,1,0]]
```

## Constraints

- `indices` is a non-negative integer tensor; exactly one 1 per row
