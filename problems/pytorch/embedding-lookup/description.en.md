# Embedding Lookup (with padding_idx Zeroing)

## Problem

An Embedding layer maps discrete integer indices to dense vectors: essentially **selecting rows** from the weight matrix $W \in \mathbb{R}^{V \times D}$ by index. By convention, the output at padding positions (`padding_idx`) is forced to an all-zero vector.

$$ \text{out}[t] = \begin{cases} \mathbf{0} & \text{indices}[t] = \text{padding\_idx} \\ W[\text{indices}[t]] & \text{otherwise} \end{cases} $$

## Rules

- Implement `solve(weight, indices, padding_idx=None)`, where `weight` has shape `(V, D)`, `indices` is an integer tensor (arbitrary shape), and the output shape is `(*indices.shape, D)`.
- Without `padding_idx`, this is equivalent to row-wise selection `weight[indices]`.
- When `padding_idx` is given, positions whose index equals it have their **entire output row set to zero** (this is the forward convention of this problem; note that `F.embedding`'s `padding_idx` only affects initialization and gradients—the forward pass does not zero it—so this problem sets it manually per this convention).
- The output dtype follows `weight`.

## Examples

```
Input: weight=arange(12).reshape(4,3), indices=[0,3,1,3], padding_idx=3
Output: [[0,1,2], [0,0,0], [3,4,5], [0,0,0]]   # both occurrences of index 3 are zeroed
```

## Constraints

- `indices` values lie in $[0, V)$; float32/float64 weights
