# PEFT: FRoD Rotation with Diagonal plus Sparse Delta

## Problem

**FRoD** (Frequency-domain Rotational Decomposition, AAAI 2026, HF PEFT v0.20.0 `tuners/frod`) packs the delta weights of same-class layers into a **shared orthogonal basis**: a joint factorization over concatenated layer weights yields a square rotation \(V\), and each layer keeps only its own diagonal coefficients \(\lambda\) (a rank-1 view) plus an **extremely sparse off-diagonal matrix \(L\)**, with \(\Delta W = U\,(S+L)\,V^{\top} - W\) (pre-correction back to the original weight). Compared to LoRA's per-layer \(A,B\), the trainable surface shrinks from \(O(r\cdot(in+out))\) to "diagonal + \(s\) sparse slots".

The deterministic chain in the official `layer.py` (anchors fully re-computable offline):

```
z   = x @ V                      # rotate into the shared basis
z_S = z @ S.T                    # sparse branch (S is not symmetric; official _sparse_activation_mm is S @ zᵀ transposed back)
z_L = z * λ                      # diagonal branch (elementwise)
out = base(x) + (z_S + z_L) @ U.T
```

Three key conventions:

1. **Shared projection \(V\)**: same-class \(W\)'s are concatenated along rows into \((\sum out, in)\) — more rows than columns takes the SVD path (\(V=V_{vh}\), row-orthogonal); fewer rows than columns takes QR + ridge averaging + `eigh` (assembled from eigenvectors of \(R^{\top}\)). Computed in float32.
2. **Sparse COO**: the \(in^2\) grid **minus the diagonal** is flattened, and `prng.randperm` picks the first `nnz = max(1, floor(in²·sparse_rate))` positions as `frod_s_indices`; values start at zero.
3. **Per-layer bookkeeping**: \(U\,\mathrm{diag}(\lambda) = W V^{\top}\) (square case via `solve(V, Wᵀ)`, non-square falls back to `pinv`), \(\lambda\) = the **column norms** of \(WV^{\top}\), \(U\) column-normalized; once \(S+L\) is loaded, `get_delta_weight` returns \(\Delta W = U(S+L)V^{\top}-W\).

**When \(S=0\), \(\Delta W \equiv 0\) exactly** (\(U\,\mathrm{diag}(\lambda)V^{\top} = W\) is an identity reconstruction), which the tests use to verify the zero-start semantics of the factorization-domain parameterization.

## Rules

- `mode="shared_projection"`: takes the list of \(W\)'s, returns the shared \(V\) (\((in,in)\), independent of layer count; the thin-matrix SVD path returns a row-orthogonal \(V\)).
- `mode="sparse_indices"`: takes \(in\), `sparse_rate`, `prng_seed`, returns `nnz×2` COO indices (`nnz = max(1, floor(in²·rate))`, off-diagonal, deterministic).
- `mode="sparse_dense"`: values + indices + `s_size`, scatters back to a dense \((in,in)\) (missing positions are 0).
- `mode="solve_u_lambda"`: takes \(W\) and \(V\), returns `{"u", "lambda"}` (\(\lambda\) is the \((in,)\) column norms).
- `mode="forward_delta"`: returns \(x\,\Delta W^{\top}\) (cross-checked against \(xV(S+L)^{\top}U^{\top}\)).
- `mode="get_delta_weight"`: returns \(U(S+L)V^{\top}-W\).
- All torch: V / sparse indices float32 (official path), W / x / ΔW float64.

## Example

\(in=6\), two layers \((5,6),(4,6)\):

```
shared_projection([W1, W2]) -> V (6,6)        # 15+8=23 rows > 6 cols → wide path
sparse_indices(6, 0.1, seed=0) -> 3 off-diagonal (row,col)
get_delta_weight(U, lam, s_vals, s_idx, V, W) -> U(S+diag λ)Vᵀ - W
```

## Constraints

- `s_indices` has no duplicates and no diagonal entries (\(i\ne j\)).
- Thin path: \(V V^{\top}=I\) (row-orthogonal); wide path: approximately column-orthogonal.
- `get_delta_weight` and `forward_delta` are transpose-consistent: \(x\,\Delta W^{\top} = xV(S+L)^{\top}U^{\top}\).
- With an ill-conditioned \(V\) (large condition number) the square path uses `solve` rather than `pinv` (the latter's rtol truncates small singular directions, numerically distinguishable).
