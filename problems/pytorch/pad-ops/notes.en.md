# Constant Padding Explained

## Definition

Pad elements at both ends of each tensor dimension. The order of `F.pad`'s `padding` tuple is **the biggest trap in this level**:

> **From the last dimension backwards**, each pair of numbers is `(left, right)`.

That is, `padding=(pl, pr, pt, pb)` corresponds to `(W left, W right, H top, H bottom)`, and if there is also a D dimension it continues backwards. Dimensions beyond the number of entries are not padded by default. This differs from `Conv2d(padding=)`, which applies "the same on every dimension".

## Coordinate mapping of the reference solution

```
for i in range(len(padding)//2):
    d = x.dim() - 1 - i        # from the last dimension backwards
    lefts[d], rights[d] = padding[2i], padding[2i+1]
```

Each output dimension has length `s + left + right`; lay the base with `torch.full(value)`, then write the original `x` into the central window `slice(l, l+s)`.

## Mode differences (this level is constant only)

- `constant`: fill with the given `value` (default 0). Convolution inputs commonly use 0.
- `reflect` / `replicate`: mirror/copy the edges, avoiding zero artifacts in style transfer or edge-sensitive cases.
- `circular`: circular padding.
This level specifies that `mode != "constant"` directly `raise ValueError` — do not use `assert` (it would be mistakenly caught by the test's `except AssertionError`); raise `ValueError`.

## Pitfalls

1. The padding tuple length must be even and no greater than `2*ndim`.
2. Writing the order as "first dimension first" or mixing it up with convolution will reverse rows and columns entirely.
3. `value` must match the dtype (giving a fractional value to an integer tensor truncates it).
