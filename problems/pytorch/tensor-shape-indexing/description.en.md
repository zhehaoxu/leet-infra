# Tensor Shape and Indexing Operations

## Problem

Half of day-to-day PyTorch work goes into "arranging a tensor into the shape you want". This problem covers slicing, `contiguous`, concatenation and reshape — the highest-frequency operations — plus one classic trap: **a view produced by slicing has non-contiguous memory, and passing it straight to a downstream operation that requires contiguous memory goes wrong**.

## Rules

Implement `solve(x: torch.Tensor, op: str) -> torch.Tensor`, where `x` is a 2-D float tensor, and return according to `op`:

| op | Semantics |
|---|---|
| `"flip"` | reverse the last dimension (e.g. `[[1,2,3]] → [[3,2,1]]`); **the returned tensor must satisfy `is_contiguous()`** |
| `"narrow"` | take the first half of the columns (column count floored; `4` columns → `2` columns) |
| `"concat"` | concatenate `x` and `x + 1` along the column direction (dim=1), doubling the number of columns |
| `"reshape"` | flatten in row-major order, then rearrange into shape `(x.shape[1], x.shape[0])` |

Raise `ValueError` when `op` is not in the table above.

> Hint: `x[:, ::-1]` is a negative-stride view whose `is_contiguous()` is `False`; only `torch.flip` or `.contiguous()` gives you contiguous memory.

## Examples

```
Input: x = [[1.0, 2.0, 3.0],
           [4.0, 5.0, 6.0]],  op = "flip"
Output: [[3.0, 2.0, 1.0],
       [6.0, 5.0, 4.0]]
```

## Constraints

- `x` has shape `(B, D)`, float32
- `D >= 2` for `"narrow"`
