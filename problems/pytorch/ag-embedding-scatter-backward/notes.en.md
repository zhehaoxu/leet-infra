# Notes: A Lookup Backward Is a Sparse Reduction

## Origin

The embedding backward surfaces in two interview settings: the basic "implement the backward of `F.embedding` by hand", and the memory/performance follow-up "why does a sparse embedding gradient let the optimizer save a lot of VRAM on a huge vocabulary". The second one requires actually understanding that the gradient is nonzero only on the \(n\) rows that were hit, so a dense layout wastes \(O(V\cdot d)\).

## Scatter-Add versus Overwriting

Since \(out_i=w_{idx_i}\), \(\partial L/\partial w_r=\sum_{i:\,idx_i=r} grad\_out_i\). Three things to notice: it is a **sum**, so any `index_put_(..., accumulate=False)` keeps only the last write per duplicate id; untouched rows act like constants in the forward and get zero; and `padding_idx` is special because the forward pins it to a zero vector that is never updated, so its row must be cleared even when `idx` contains it. With `scatter_add_` the `src` must be broadcast to the output shape, or simply use `index_add_(0, idx, grad)`, which handles duplicates natively.

## The Correct Order for scale_grad_by_freq

PyTorch's semantics are "divide each token's contribution by the occurrences of that id": \(\partial L/\partial w_r=\sum_{i:\,idx_i=r} grad\_out_i/f_{idx_i}\) with \(f_r\) the count of \(r\) in `idx`. Divide first, then accumulate; the total contribution of a repeated token therefore equals the average of its \(grad\_out\) rows, not the summed row divided again by \(f_r\). The two spellings differ by a factor of \(f_r\) and show up immediately in a cross-check.

## Core Ideas and Pitfalls

- **Sparse branch**: after `torch.sparse_coo_tensor(indices, values, size)` you must call `coalesce()`; judging checks `is_coalesced`, the layout, and `values().shape[0] <= n` so the result cannot silently blow up to \(V\) entries.
- **`padding_idx` together with `scale_grad_by_freq`**: dilute by frequency first, then zero the padding row.
- **Range checks**: negative `idx` must not slip through via Python's negative-index semantics — raise ValueError explicitly.
- **dtype**: accumulate in float64 and return the dtype of `grad_out`, otherwise float32 cases drift an order of magnitude away from autograd.

## How It Differs from Neighboring Problems

`ag-pooling-backward` also routes gradients back to their origin, but the destination is decided by a `max` comparison; here the destinations come from an external `idx`, and rows additionally merge several tokens. In `ag-conv2d-backward` the scatter happens over spatial windows.

## Cross-Check Basis and Complexity

The oracle builds `weight = torch.empty(V,d).uniform_()`, calls `F.embedding(weight, idx, padding_idx=..., scale_grad_by_freq=..., sparse=...)` and takes `torch.autograd.grad`, atol=1e-9 in float64. The example uses only the three rows \([2,4],[6,8],[10,12]\), covering a duplicate row, a hit row and an empty row at once. Complexity is \(O(nd)\) for the accumulation plus \(O(n\log n)\) to coalesce; a dense formulation still pays an \(O(Vd)\) zeroing cost.
