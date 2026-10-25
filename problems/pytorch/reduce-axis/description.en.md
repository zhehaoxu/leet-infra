# Sum Reduction along an Axis

## Problem

A reduction "squashes" one axis, combining the data along it into a single value with an aggregation function (here: summation).
Understanding "which axis is reduced and which axes remain" is the foundation of reading and writing every tensor operation (softmax, attention, BatchNorm, loss functions).
Please implement summation along an arbitrary axis by slicing and accumulating, **without using `torch.sum`**.

## Rules

- Implement `solve(x, axis, keepdim=False)`, returning the tensor summed along `axis`.
- `axis` may be negative (-1 means the last dimension).
- With `keepdim=False` the axis is removed; with `keepdim=True` it is kept with length 1.
- The result must match `torch.sum(x, dim=axis, keepdim=keepdim)`.

## Examples

```
Input: x = [[1, 2, 3],
          [4, 5, 6]]
  axis=1 (along columns, collapsing each row to a scalar) → [6, 15]
  axis=0 (along rows, collapsing each column to a scalar) → [5, 7, 9]
  axis=1, keepdim=True             → [[6], [15]]
```

## Constraints

- Support inputs of any dimensionality; sum along `axis`
- Built-in reduction operators such as `torch.sum / .sum() / .mean()` are forbidden (accumulate element by element via slicing)
