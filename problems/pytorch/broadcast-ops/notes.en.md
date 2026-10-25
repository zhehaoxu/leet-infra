# Broadcasting in Practice Explained

## Broadcasting rules (in two sentences)

Alignment compares dimensions one by one **from the last dimension backwards**: if the two dimensions are equal, or one of them is 1, they can broadcast; otherwise it errors. Missing leading dimensions are padded with 1. The result is that a size-1 dimension is "virtually replicated" to the other operand's size — **no real copy happens**, and the stride along that dimension is set to 0.

## Three typical patterns

| op | reference solution | semantics |
|----|--------|------|
| `row_scale` | `a * b[:, None]` | `b` is `(B,)`; `[:, None]`→`(B,1)`, scaling each **column** row by row. |
| `col_bias` | `a + b[None, :]` | `b` is `(D,)`; `[None, :]`→`(1,D)`, adding a per-column bias to each **row**. |
| `outer` | `a[:, None] * b[None, :]` | two 1D vectors → an `(len(a), len(b))` outer-product matrix. |

## The coordinate sense of unsqueeze

`x[:, None]` inserts a length-1 axis at dim=1, turning a 1D vector into a 2D column vector; `x[None, :]` inserts an axis at dim=0 to make a row vector. We write `b[:, None]` rather than plain `b` because `a` is 2D `(B,D)`: without the inserted axis, `(B,D)` and `(B,)` fail to align when the last dimension $D\ne B$. **First decide along which axis you want `b` spread out, then add `None`/`unsqueeze` at that position.**

## Common pitfalls

- Broadcast tensors are read-only-semantics "expanded views"; to materialize an independent copy use `.expand(...).contiguous()` or `.repeat(...)`.
- `b[:, None]` and `b.unsqueeze(1)` are completely equivalent; pick whichever reads better.
- Broadcasting checks shapes, not values; mismatched sizes raise `RuntimeError` rather than silently truncating.

## Related

linear-layer (the $b$ in $xW^\top+b$ is a col_bias broadcast), the affine step of layernorm/batchnorm, and outer-product-style attention are all built on these three patterns.
