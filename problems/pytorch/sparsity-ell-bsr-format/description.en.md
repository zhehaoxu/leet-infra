# Sparsification: ELL / BSR Conversion with Storage Accounting

## Problem

The same sparse matrix is stored in different "shapes" on different hardware:

- **CSR (compressed sparse row)**: compact, but rows have unequal lengths, so thousands of GPU threads walking rows in parallel diverge badly (uneven lengths → idle lanes in a warp).
- **ELL (equal-length rows)**: every row is padded to \(M\), the maximum non-zero count over rows, and stored as two dense \(R\times M\) arrays — `values` and column indices. Rows are uniform and accesses perfectly regular, at the cost of wasted padding slots; the more skewed the row lengths, the worse the waste.
- **BSR (block sparse row)**: stores \(b_r\times b_c\) tiles and keeps only tiles with at least one non-zero. It retains compression while every tile is a contiguous, aligned dense mini-matrix — directly consumable by Tensor Cores / tiled GEMM.

In practice the format choice is decided by a storage bill: for a given matrix, three quantities — the NNZ count, the longest row length \(M\), and the non-zero block count \(NB\) — determine the byte cost of each format. This problem asks you to convert a dense matrix into ELL and BSR at the same time, and to compute the byte cost of all three formats.

## Rules

Given `dense` (float32, \((R,C)\), \(R\%b_r=0\), \(C\%b_c=0\), non-zero meaning `!= 0.0`, with \(-0.0\) treated as zero) and the block shape `block=(b_r,b_c)`:

**ELL**

1. Let \(M=\max_i(\text{non-zero count of row } i)\) (for an all-zero matrix \(M=0\), so both arrays are \(R\times0\));
2. each row stores its non-zeros in **ascending column** order starting from slot 0;
3. pad the tail: `values` with `0.0`, `indices` with `-1` (so padding is distinguishable from column 0);
4. `ell_values` is float32 \((R,M)\), `ell_indices` is int64 \((R,M)\).

**BSR**

1. split the matrix into an \(R/b_r\times C/b_c\) grid of \(b_r\times b_c\) tiles;
2. keep a tile if **at least one** element in it is non-zero, drop it otherwise;
3. emit in block-row-major order (block rows first, block columns ascending): `bsr_values` (\((NB,b_r\cdot b_c)\), each tile flattened row-major), `bsr_indices` (block column of each tile, int64), `bsr_indptr` (length \(R/b_r+1\), cumulative tile counts before each block row);
4. with `block=(1,1)` BSR degenerates into "COO laid out in row-major order" and \(NB=\text{nnz}\).

**Storage bytes**

Let \(\text{nnz}\) be the total non-zero count, \(M\) the longest row non-zero count, \(NB\) the non-zero block count, \(N_r=R/b_r\); values cost 4 bytes and indices 8 bytes:

$$
\text{CSR} = \underbrace{4\,\text{nnz}}_{\text{values}} + \underbrace{8\,\text{nnz}}_{\text{col idx}} + \underbrace{8(R+1)}_{\text{indptr}}
$$

$$
\text{ELL} = 4RM + 8RM
\qquad
\text{BSR} = 4\,NB\,b_r b_c + 8\,NB + 8(N_r+1)
$$

Return `nbytes` (int64, length 3) as `[CSR, ELL, BSR]`. `torch.sparse` / `to_sparse_csr` / `to_sparse_bsr` are forbidden.

## Example

`dense` is 4×4 with `block=(2,2)`:

```
dense = [[1, 0, 0, 2],
         [0, 0, 3, 0],
         [4, 5, 0, 0],
         [0, 0, 0, 6]]

row non-zero counts = [2, 1, 2, 1] -> M = 2

ELL:
  row0 cols [0,3] -> values [1,2]   indices [0, 3]
  row1 cols [2]   -> values [3,0]   indices [2,-1]   # 1 padded slot
  row2 cols [0,1] -> values [4,5]   indices [0, 1]
  row3 cols [3]   -> values [6,0]   indices [3,-1]   # 1 padded slot

BSR (2x2 grid of 2x2 tiles, all four tiles non-zero):
  tile(0,0)=[[1,0],[0,0]] -> [1,0,0,0]  block col 0
  tile(0,1)=[[0,2],[3,0]] -> [0,2,3,0]  block col 1
  tile(1,0)=[[4,5],[0,0]] -> [4,5,0,0]  block col 0
  tile(1,1)=[[0,0],[0,6]] -> [0,0,0,6]  block col 1
  bsr_indices = [0, 1, 0, 1], bsr_indptr = [0, 2, 4]
```

`nnz=6`, `M=2`, `NB=4`, `N_r=2`, hence

```
CSR = 6*4 + 6*8 + (4+1)*8   = 24 + 48 + 40 = 112
ELL = 4*4*2 + 8*4*2         = 32 + 64       =  96
BSR = 4*4*4 + 8*4 + (2+1)*8 = 64 + 32 + 24  = 120
nbytes = [112, 96, 120]
```

## Constraints

- CPU float32 input; `ell_values`/`bsr_values` are float32, all indices and byte counts int64.
- ELL row width is always \(M\): an all-zero row must be padded for its full width with `-1`/`0.0`, never skipped.
- Padding is always "value 0, index −1" and must follow the real elements (ascending columns).
- BSR keeps only tiles with at least one non-zero; a **partially** non-zero tile is stored whole (its zeros included).
- All-zero matrix: \(M=0\), \(NB=0\), `indptr` all zeros.
- ELL or BSR must reconstruct the original matrix losslessly.
