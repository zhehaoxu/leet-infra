# KV Cache: WKVD Least-Squares Merging of Evicted Values

## Problem

When KV cache compression reaches the point where tokens must be dropped, the usual choices are "drop the oldest" or "drop the lowest score", but discarding a value vector outright erases all information carried by that key. The observation behind the WKVD family is that **an evicted value vector can usually be reconstructed approximately from a linear combination of the surviving values**—the value space of an attention head is low-rank and highly redundant. If it can be reconstructed, it need not be dropped: **project it into the subspace spanned by the survivors and merge it in**.

Formally, let \(V_{\text{keep}}\in\mathbb R^{K\times d}\) hold the surviving values (one row per token) and \(V_{\text{evict}}\in\mathbb R^{E\times d}\) the evicted ones. We look for a coefficient matrix \(C\in\mathbb R^{K\times E}\) with

\[
C^\*=\arg\min_{C}\ \big\|V_{\text{keep}}^{\top}C - V_{\text{evict}}^{\top}\big\|_F
\]

and then add the evicted information into the surviving rows:

\[
V' = V_{\text{keep}} + C\,V_{\text{evict}} \in \mathbb R^{K\times d},
\qquad \text{residual} = \big\|V_{\text{keep}}^{\top}C - V_{\text{evict}}^{\top}\big\|_F
\]

Note the least-squares is **column-wise**: \(V_{\text{keep}}^\top\) is \((d\times K)\) and \(V_{\text{evict}}^\top\) is \((d\times E)\), so \(C\) has \(K\times E\) unknowns and each column independently solves "fit one evicted vector with the \(K\) basis vectors".

Reading \(V'\) row by row is intuitive: \(V'_i = V_{\text{keep},i} + \sum_{e} C_{i,e}\,V_{\text{evict},e}\)—surviving token \(i\) absorbs a share of the evicted tokens.

Three properties drive the checks in this problem:

1. **A full basis gives zero residual.** If \(K\) covers the spanned dimension, every evicted vector is exactly representable, the residual is 0, and the merge is lossless.
2. **Orthogonal directions cannot be represented.** When the component lies in the orthogonal complement of the row space of \(V_{\text{keep}}\), the coefficients must be 0, the merge leaves the survivors unchanged, and the residual equals the norm of that component.
3. **Rank deficiency demands a pseudo-inverse.** Rows of \(V_{\text{keep}}\) may be linearly dependent (common in low-rank heads); then \(V_{\text{keep}}^\top\) is column-rank-deficient and the normal equations have no unique solution. You must use the Moore–Penrose pseudo-inverse to obtain the **minimum-norm solution** rather than erroring out or returning exploding values.

## Task

Implement `solve(V_keep, V_evict)`:

- Input: `V_keep` (float32 `(K, d)`), `V_evict` (float32 `(E, d)`), with \(K,E\ge 1\).
- Output `(merged, coef, residual)`: `merged` is `(K, d)` float32; `coef` is `(K, E)` float32; `residual` is a 0-dim float32 scalar (the Frobenius residual \(\|\cdot\|_F\)).
- Solve internally in float64 and cast the outputs back to float32.
- Raise `ValueError` when `d` differs, when `K`/`E` is 0, or when an input is not 2-D.
- Do not regularize \(V_{\text{keep}}\) and do not require full rank—degenerate cases (linearly dependent rows, \(K>d\)) must still return finite values.

## Example

```
V_keep  = [[1, 0],
           [0, 1]]            # standard orthonormal basis, K = d = 2
V_evict = [[0.5, 0.25]]       # E = 1

basis  = V_keep^T = [[1,0],[0,1]]
target = V_evict^T = [[0.5],[0.25]]
coef   = pinv(basis) @ target = [[0.5],[0.25]]      # full basis, read off the components

merged = V_keep + coef @ V_evict
       = [[1, 0], [0, 1]] + [[0.5],[0.25]] @ [[0.5, 0.25]]
       = [[1, 0], [0, 1]] + [[0.25, 0.125],
                             [0.125, 0.0625]]
       = [[1.25,  0.125],
          [0.125, 1.0625]]

residual = ||basis @ coef - target||_F = 0
```

The evicted `[0.5, 0.25]` is split into `0.5×row0 + 0.25×row1`; those two shares are added to the two surviving rows (an outer product), so every entry of the merge gains a little.

## Constraints

- **The least-squares direction is along columns**: solve \(V_{\text{keep}}^\top C \approx V_{\text{evict}}^\top\), not \(V_{\text{keep}} W \approx V_{\text{evict}}\). The former yields \((K\times E)\) coefficients and the merge adds rows; the latter does not even type-check (\(K\times d\) times \(d\times d\) cannot produce \(E\times d\)).
- **The residual is the solve residual, not a merge error**: \(\|V_{\text{keep}}^\top C - V_{\text{evict}}^\top\|_F\) measures how much of the evicted information is representable; the merged \(V'\) and \(V_{\text{evict}}\) are not directly comparable (different shapes).
- **Prefer the pseudo-inverse over inverse or a failing solve**: `pinv` returns the minimum-norm solution under rank deficiency, while `torch.linalg.inv` blows up and `solve` yields inf/nan.
- **The merge increment is bilinear**: \(V'-V_{\text{keep}} = C\,V_{\text{evict}}\); scaling \(V_{\text{evict}}\) by 2 scales the coefficients by 2, hence the increment by 4—not by 2.
- Compute coefficients and the residual in float64 before downcasting, otherwise `pinv` on near-singular matrices in float32 returns noise.
