# Top-K Selection

## Problem

Top-K picks the `k` largest values along some axis together with their original indices, sorted in
descending order. It is the common primitive behind beam search, expert routing (each token in MoE selects
top-k experts), retrieval recall in recommenders, nucleus/top-k sampling, and more. Please implement it
**without `torch.topk`**, using the selection method "repeatedly take the maximum, then mask it out".

## Rules

- Implement `solve(x, k)`; take the top-k along the **last dimension** and return a `(values, indices)` tuple.
- `values` has shape `(..., k)` and is arranged **in descending order**; `indices` are their original
  positions along the last dimension.
- Each round, take the current maximum, record its value and index, then remove that position from the
  candidates (set it to `-inf`); repeat `k` times.
- The result must agree with `torch.topk(x, k, dim=-1, sorted=True)` on the test data (no duplicate values).

## Example

```
Input: x = [[3, 1, 7, 2, 9],
          [5, 8, 0, 6, 4]], k = 3
Output: values  = [[9, 7, 3],
                 [8, 6, 5]]
     indices = [[4, 2, 0],
                [1, 3, 0]]
```

## Constraints

- float32 input; `1 <= k <=` length of the last dimension
- The tests use random continuous values (no ties), so the top-k indices are uniquely determined
