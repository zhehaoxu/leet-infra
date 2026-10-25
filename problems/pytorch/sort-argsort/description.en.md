# Sort & Argsort

## Problem

Sorting gives the **values** of a row of elements after ascending reorder; `argsort` gives the **original indices** needed to arrange those values in ascending order.
The two often appear together: the indices from `argsort` can drive a `gather` that reorders associated tensors (e.g. sorting bounding boxes by score,
sorting the sequences in a batch by length). Please implement it with selection sort, **without `torch.sort/argsort`**.

## Rules

- Implement `solve(x)`, sorting **ascending along the last dimension**, returning the tuple `(values, indices)`.
- `values`: the values after ascending sort, same shape as the input; `indices`: the positions of the corresponding elements in the original last dimension.
- Build the result step by step with the selection method: "take the minimum of the unsorted part, write out the value and index, then remove it from the candidates (set it to `+inf`)".
- The result must agree with `torch.sort(x, dim=-1)` on the test data (no duplicate values).

## Examples

```
Input: x = [[3, 1, 7, 2, 9],
          [5, 8, 0, 6, 4]]
Output: values  = [[1, 2, 3, 7, 9],
                 [0, 4, 5, 6, 8]]
     indices = [[1, 3, 0, 2, 4],
                [2, 4, 0, 3, 1]]
```

## Constraints

- float32 inputs; sort along the last dimension
- Tests use random continuous values (no ties), so the sorted result and the argsort indices are uniquely determined
