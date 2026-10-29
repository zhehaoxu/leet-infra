# Deep Dive: ELL / BSR Conversion with Storage Accounting

## Origin
ELL is the classic format sparse BLAS uses for "equal-length rows, regularly parallelisable" workloads (the `csrmm`/`hyb` family and the ELL paths in cuSPARSE/cuBLAS); BSR is the block format of `scipy.sparse.bsr_matrix`, `torch.sparse_bsr_tensor` and cuSPARSE `bsrmm`, and the landing point for tiled GEMM and Tensor Core sparse acceleration. This problem lines them up against CSR and focuses on the storage bill: for one matrix, the space cost of the three layouts is fixed by nnz, the longest row \(M\), and the non-zero block count \(NB\). It complements `sparsity-csr-encode` in this collection (CSR encoding only) — that one covers *how to compress*, this one covers *what the compressed form looks like and whether it pays off*.

## Key points and pitfalls
1. **ELL's \(M\) is global**: the maximum non-zero count over all rows, not a per-row length, so short rows are padded up to \(M\). An all-zero matrix gives \(M=0\) and arrays of shape \((R,0)\), not \((R,1)\).
2. **Padding indices must be −1**: padding values with 0 may look harmless (a real non-zero is never 0), but column 0 is a legal value, so empty slots need −1 for a decoder to tell them apart.
3. **Columns ascending within a row**: `nonzero` / a stable `argsort((~nz))` naturally yields ascending order with non-zeros first; take the first \(M\) entries.
4. **BSR's criterion is "any non-zero in the tile", but the whole tile is stored**: a partially non-zero tile keeps its zeros; dropping them would break the row-major tile flattening.
5. **`bsr_indptr` is a block-row pointer**: length \(R/b_r+1\), each entry adding that block row's non-zero tile count; using an element-level `indptr` or omitting the final slot is wrong.
6. **Tile flattening is row-major**: `reshape(nr, br, nc, bc).permute(0,2,1,3).reshape(nr, nc, br*bc)` is the canonical form; a plain `reshape(nr,br,nc,bc)` gives column-major tiles.
7. **Each format's byte bill uses its own granularity**: CSR counts nnz elements, ELL stores \(R\cdot M\) slots (padding included), BSR stores \(NB\cdot b_r b_c\) values (in-tile zeros included). The denominators differ and cannot be mixed.
8. **`block=(1,1)` is the degenerate check**: then \(NB=\) nnz and each tile holds one element, so BSR must equal row-major COO.

## Verification
The reference solution takes per-row non-zero columns via a stable `argsort` on `~nz` for ELL, flattens tiles with `reshape+permute+reshape`, finds non-zero tiles with `any(dim=2)`, builds `indptr` with `cumsum`, and evaluates the three byte formulas directly. The judge `_oracle` uses pure-Python double loops over rows and tiles with `!= 0.0`, assembling all six return values by hand, then compares item by item. Additional assertions: independent ELL and BSR decoders must reproduce the dense matrix exactly; a padded example has shape \((3,3)\) and indices exactly `[[0,-1,-1],[-1,-1,-1],[0,1,2]]`; the all-zero matrix degenerates (\(M=0\), \(NB=0\), `indptr=[0,0,0]`, CSR reduced to 40 pointer bytes); a fully dense matrix has no padding (every row full, `ei>=0`); with `block=(1,1)` the `bsr_indices`/`bsr_indptr` match row-major order.

## Complexity
\(O(RC)\) scanning; ELL emits \(R\cdot M\) values plus indices, BSR emits \(NB\cdot b_r b_c\) values, and the byte bill is \(O(1)\).
