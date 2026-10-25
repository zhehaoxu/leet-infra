# Tensor Shape and Indexing Operations in Depth

## What this problem trains

Four high-frequency "reshape / slice" operations that tie together the three things easiest to get wrong in PyTorch: **view vs copy, stride, and dimension alignment**. `op` is dispatched from a string, and an invalid value raises `ValueError`.

## Per-operation semantics

| op | Reference solution | Key point |
|----|--------|------|
| `flip` | `torch.flip(x, dims=[-1])` | Reverse the last dimension. The slice `x[:, ::-1]` does not support negative strides in this version; moreover a slice yields a **negative-stride view**, and many downstream operators require contiguous memory. `torch.flip` directly produces a new contiguous tensor. |
| `narrow` | `x[:, : x.shape[1] // 2]` | Take the first half along the columns (width divided by 2 with integer division). This is a "narrowed view": zero copy, sharing storage with the original tensor. |
| `concat` | `torch.cat([x, x + 1], dim=1)` | Concatenate `x` and `x+1` along dim=1, doubling the column count. `cat` allocates a new tensor and copies. |
| `reshape` | `x.reshape(x.shape[1], x.shape[0])` | Swap the sizes of the first two dimensions to get a rearrangement with the same total number of elements (not a transpose; it only looks like one for a square matrix). It may return a view or a copy, depending on the memory layout. |

## The view trap in flip / narrow / reshape

- **narrow / reshape usually give views**: modifying them reaches through and affects the original tensor; flip gives a copy. When you need an independent copy, use `.contiguous()` / `.clone()`.
- **A non-contiguous tensor from transpose cannot be `reshape`d directly**: `x.transpose(...).reshape(...)` copies implicitly; to avoid hidden copies use `.view()`, which instead raises an error as a warning.
- When a dimension is uncertain, use `x.reshape(-1, ...)` and let that dimension be inferred.

## Invalid op

You must `raise ValueError` — the judge uses `pytest.raises` to check the negative path; returning None or not raising both fail.

## Related

broadcast-ops explains alignment with unsqueeze; concat-stack-split is the family of this problem's `cat`; the stride/contiguous concepts carry over to problems such as grid-sample and unfold.
