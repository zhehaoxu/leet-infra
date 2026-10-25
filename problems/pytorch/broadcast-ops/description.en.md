# Broadcasting in Practice

## Problem

Broadcasting lets tensors of different shapes be used directly in element-wise operations: dimensions are aligned from the trailing end, and dimensions that are `1` or missing are expanded automatically. It is the number-one productivity tool for vectorized PyTorch code — used well, broadcasting eliminates the vast majority of Python loops.

## Rules

Implement `solve(a: torch.Tensor, b: torch.Tensor, op: str) -> torch.Tensor`, using only broadcast operations (`unsqueeze` / `None` indexing / the four arithmetic operators); writing element-wise Python loops is not allowed:

| op | inputs | semantics | output shape |
|---|---|---|---|
| `"row_scale"` | `a (B,D)`, `b (B,)` | multiply row `i` by `b[i]` | `(B, D)` |
| `"col_bias"` | `a (B,D)`, `b (D,)` | add `b[j]` to column `j` | `(B, D)` |
| `"outer"` | `a (m,)`, `b (n,)` | outer product `a[i] * b[j]` | `(m, n)` |

When `op` is not in the table above, raise `ValueError`.

> Hint: before operating on a 2D tensor with a 1D tensor, you usually need `a[:, None]` (equivalent to `a.unsqueeze(1)`) to turn the length-`B` vector into `(B,1)` so that broadcasting aligns it across each column.

## Examples

```
a = [1.0, 2.0], b = [10.0, 20.0, 30.0], op = "outer"
Output: [[10.0, 20.0, 30.0],
       [20.0, 40.0, 60.0]]
```

## Constraints

- Inputs are float32 tensors and dimension lengths are guaranteed to match the shapes in the table
- The result must match the vectorized reference implementation element-wise (at the `allclose` level)
