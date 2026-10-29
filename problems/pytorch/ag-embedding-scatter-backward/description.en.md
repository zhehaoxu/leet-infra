# Embedding Backward: Duplicate id Accumulation, padding_idx Rows and Sparse Grad

## Problem Description

The forward of `F.embedding` is a lookup, but its backward is a **scatter-add**: when one token id appears several times in a batch, its gradient must sum the corresponding rows of `grad_out` rather than let the last write overwrite the earlier ones. Add the three engineering conventions — the `padding_idx` row must never receive gradient, `scale_grad_by_freq` divides by the term frequency, and `sparse=True` must hand back a genuinely coalesced COO tensor — and a "lookup table" turns into a sparse reduction. Given `grad_out, idx, num_embeddings` plus the three switches, return a gradient of shape `(num_embeddings, dim)`.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- Row \(r\) of the output is the sum of every `grad_out[i]` with `idx[i] == r`; untouched rows stay zero.
- The `padding_idx` row is **zeroed entirely** (it is a constant in the forward) and this takes precedence over the other rules.
- With `scale_grad_by_freq=True`, each token's gradient is first divided by the number of occurrences of that id in `idx` before being accumulated.
- With `sparse=True` return a `torch.sparse_coo_tensor` that has been `coalesce()`d, whose `to_dense()` matches the dense path entry by entry.
- Out-of-range `idx`, a non-2-D `grad_out`, a row-count mismatch with `idx`, and `num_embeddings <= 0` all raise ValueError; the return dtype matches `grad_out`.

## Example

Vocabulary of 4 rows, dim 2, `idx=[1,1,0]`; numbers come from the EXAMPLE_* constants in test_cases.py:

```
grad_out = [[2, 4], [6, 8], [10, 12]]    idx = [1, 1, 0]
dense    = [[10, 12],   # row 0 comes from the third token
            [ 8, 12],   # row 1 = [2,4] + [6,8] (accumulated, not overwritten)
            [ 0,  0], [ 0,  0]]
freq     = [[10, 12], [4, 6], [0, 0], [0, 0]]   # scale_grad_by_freq=True: id 1 occurs twice
```

## Constraints

- Accumulate in float64 then cast back to `grad_out.dtype`; cross-checked against `F.embedding` plus `torch.autograd.grad` with atol=1e-9 in float64
- `idx` is 1-D int64 with `n` up to a few thousand; the sparse path is verified with `values().shape[0] <= n` so it cannot silently degrade to dense
