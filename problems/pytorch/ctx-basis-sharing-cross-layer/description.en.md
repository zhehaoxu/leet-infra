# PEFT: BEFT Bypass-Bias Fine-Tuning

## Problem

**Basis Sharing** (Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression, arXiv:2410.03765, ICLR 2025, official repo `TUDa-HWAI/Basis_Sharing`) goes beyond per-layer independent SVD compression: the same-name projection weights of **different layers are concatenated into one wide matrix and factorized jointly** — every layer in the group reuses one set of shared basis vectors, and each layer keeps only its own narrow coefficient matrix. Parameters drop from \(m \cdot out \cdot in\) to \(in \cdot k + m \cdot k \cdot out\), outperforming per-layer SVD and parameter-sharing baselines at large compression ratios.

The official compression trunk (`group.py: Group._init_basis_coefficient`, llama path transposes the weights first):

```
W_cat = [W_1.T | W_2.T | ... ]             # (in, m·out) wide matrix
S    = chol(H).T                           # H is the calibration Hessian (SPD)
w    = S @ W_cat                           # SVD-LLM whitening
u, sigma, v = svd(w)                       # thin SVD
basis = invs @ u[:, :k] @ diag(sigma[:k])  # (in, k) shared basis
C_l   = v.T[:, seg_l][:k, :]               # (k, out_l) per-layer coefficients
```

Reconstruction \( \hat W_l = (B\, C_l)^\top \); forward \(y = x\,B\,C_l\) (the Basis→Coefficient chain). The official coefficient update (`update_model`, re-solving after the basis is retrained):

```
C_cat = (W_cat.T @ H @ B) @ (B.T @ H @ B)^-1    # closed-form least squares, split per layer
```

**Key identity**: with \(H = S^\top S\) the objective equals \(\lVert S W_{cat} - U_k \Sigma_k C\rVert_F^2\), whose optimum is exactly the initial SVD coefficients — **the two stages coincide numerically under the SVD basis**; after the basis changes, LS re-solves (normal equations \(B^\top H (W - BC) = 0\)). The calibration-reuse map (same-input projections share the Hessian): q/v use k's H, gate uses up's H, everything else uses its own. SVD/Cholesky factors are explicit inputs.

## Rules

- `mode="scaled_concat"`: given per-layer \((out_l, in)\) weights and the upper-triangular Cholesky factor \(S\), return \((W_{cat},\ S\,W_{cat})\), both \((in,\ m \cdot out)\) — whitening acts on the **wide concatenated matrix**.
- `mode="build_shared_basis"`: given u/σ/invs and num_basis, return \(B = S^{-1}U_{:k}\,\mathrm{diag}(\sigma_{:k})\ (in, k)\) (top-k columns, σ descending).
- `mode="initial_coefficients"`: given \(v^\top\ (in, m \cdot out)\) and out_dim, take the per-layer column segment \([i \cdot out, (i{+}1) \cdot out)\) and its first k rows, returning \([(k, out_l)]\).
- `mode="least_squares_update"`: the closed form \((W_{cat}^\top H B)(B^\top H B)^{-1}\), transposed and split per layer.
- `mode="reconstruct"` / `mode="forward_shared"`: per-layer \((B\,C_l)^\top\) and \(x\,B\,C_l\).
- `mode="calib_source"`: the calibration-reuse map.
- float64 throughout; no decomposition is run at solve time.

## Example

\(in=8, out=6\), group of \(m=3\) layers, \(k=4\):

```
scaled_concat([W1,W2,W3], S) -> w_cat (8,18), w = S@w_cat
build_shared_basis(u, sigma, invs, 4) -> B (8,4)
initial_coefficients(v.T, 6, 4) -> [(4,6)]×3
reconstruct(B, C) -> W_hat_l (6,8)；forward_shared(x, B, C) -> x@B@C_l
```

At full rank \(k=in\), \(B_{full} C = W_{cat}\) recovers the weights exactly (\(S\) invertible); truncated reconstruction is an approximation.

## Constraints

- Full-rank cross-checks: \(B_{full}\ C_{full} = W_{cat}\) and the full-rank LS solution equals the full \(v^\top\) (two independent paths).
- Under the SVD basis the LS re-solve equals the initial coefficients (the \(H = S^\top S\) identity); on a perturbed basis LS satisfies the normal equations and differs observably from the initial coefficients.
- Forward and reconstruction share one source: \(x\,B\,C_l = x\,\hat W_l^\top\).
- The q/v→k and gate→up calibration maps are exact (same-input projections).
