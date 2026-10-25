# Sparsity: Encoding a Dense Matrix as CSR

## Problem

After pruning, the vast majority of a network's weights are 0; storing them in a dense layout wastes memory and bandwidth. **CSR (Compressed Sparse Row)** is the standard compressed storage for sparse matrices: it keeps only the nonzero elements, described by three arrays —

- `values`: all nonzero elements, in **row-major order** (row by row, column by column within each row);
- `indices`: the **column index** of each nonzero element;
- `indptr`: a row-pointer array of length \(R+1\); `indptr[i]:indptr[i+1]` is the slice of row \(i\)'s nonzero elements in `values`/`indices`, with `indptr[0]=0` and `indptr[R]=nnz`.

## Rules

Input `dense` (float32, 2-D \((R,C)\), where "nonzero" is defined as `!= 0.0`). Return the triple `(indptr, indices, values)`: `indptr` and `indices` are int64 tensors, `values` is float32. Nonzero elements are expanded strictly in row-major order. Returning `dense.to_sparse_csr()` directly is forbidden — the three arrays must be constructed by hand.

## Examples

`dense=[[1,0,0,2],[0,0,3,0],[4,5,0,0]]`:

```
Row 0 nonzeros: col0=1, col3=2
Row 1 nonzeros: col2=3
Row 2 nonzeros: col0=4, col1=5
indptr  = [0, 2, 3, 5]        # cumulative nnz at the end of each row
indices = [0, 3, 2, 0, 1]
values  = [1, 2, 3, 4, 5]
```

## Constraints

- CPU float32; `indptr` is non-decreasing, starts at 0 and ends at nnz; `indices` are column-increasing within each row (naturally produced by the row-major scan).
- All-zero matrix: `indptr=[0,0,...,0]`, `indices`/`values` empty.
- Length relations among the three arrays: `len(indptr)=R+1`, `len(indices)=len(values)=nnz`.
