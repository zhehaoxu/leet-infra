# Deep Dive: Encoding a Dense Matrix as CSR

## Origin
CSR (Compressed Sparse Row) is the general sparse-matrix format used by scipy / PyTorch `to_sparse_csr` / cuSPARSE: the three arrays `indptr/indices/values`. The real storage of pruned models and sparse GEMM are both built on top of it.

## Core Ideas and Pitfalls
1. **Row-major order**: `values`/`indices` must be expanded row by row, column by column within each row — `torch.nonzero` naturally returns them in this order.
2. **`indptr` has length R+1 and holds cumulative nnz**: `indptr[i]` is the number of nonzeros before row \(i\), and `indptr[i+1]-indptr[i]` is row \(i\)'s nnz; it starts at 0 and ends at the total nnz. The final slot is the easiest thing to miss when constructing it.
3. **Empty rows**: when a row is all zero, two adjacent `indptr` values are equal; you cannot skip them.
4. **dtype**: `indptr`/`indices` are int64, and `values` follows the original matrix's float32; the judge asserts on dtype.
5. "Nonzero" is decided strictly with `!= 0.0`; -0.0 counts as 0.

## Cross-Checking Against the Oracle
The reference solution uses `nonzero→counts→cumsum`, assembling the three arrays in vectorized fashion. The judge's `_oracle` scans element by element with a pure Python double loop and accumulates by hand (different structure, equivalent results), cross-checking `indptr/indices/values` one by one (`tolist()` equality plus `allclose` on values). Additional assertions cover "length relations len(indptr)=R+1, len(indices)=len(values)=nnz", "the all-zero matrix degeneration", and "the three arrays can reconstruct the original matrix".

## Complexity
\(O(RC)\) scanning, \(O(\text{nnz}+R)\) storage.
