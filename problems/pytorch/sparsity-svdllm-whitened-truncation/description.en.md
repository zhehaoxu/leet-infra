# SVD-LLM Whitened-Domain Truncation Compression

## Problem

Plain SVD compression has a counterintuitive flaw: **truncating smaller singular values can yield larger compression loss** (activation outliers break the link between the weight spectrum and the loss). SVD-LLM (ICLR 2025) fixes this with truncation-aware data whitening — move the weight into a whitened domain before the SVD, so that "truncate small singular values" maps directly to "low compression loss".

You are given four inputs (the Cholesky factor L of the calibration H comes from upstream profiling and is outside the judging scope):

- `weight`: the weight W to compress, shape (M, N);
- `scaling_diag_matrix`: the whitening matrix L ((N, N), lower-triangular Cholesky factor); omitted when `direct_update=True`;
- `x`: calibration input (P, N); `y`: the dense model's corresponding output (P, M).

## The pipeline to implement

### 1. Whitened-domain SVD

```
W_scale = W @ L
U, S, Vᵀ = svd(W_scale)          # full_matrices=False
```

### 2. Parameter-budget truncation

```
k = int(M·N·ratio / (M+N))        # compressed params ≈ ratio × original
k < 1 → error (no upper-bound check needed: with ratio ≤ 1, k is always < min(M,N))
```

### 3. √Σ factor split (v side carries the L⁻¹ compensation)

```
Σ_k = diag(S[:k]),  √Σ_k = sqrt(Σ_k)
svd_u = U[:, :k] @ √Σ_k
svd_v = √Σ_k @ (Vᵀ[:k, :] @ L⁻¹)   ← truncation-aware core: right-multiply L⁻¹ to undo the whitening
with direct_update=True skip whitening: svd_v = √Σ_k @ Vᵀ[:k, :] (and L is not needed)
new_w = U[:, :k] @ Σ_k @ (Vᵀ[:k, :] @ L⁻¹)
```

When L is singular the official defense adds `+1e-6·I` before inverting.

### 4. Closed-form update (layer-wise closed-form update)

```
x' = (x @ svd_v_coreᵀ) @ Σ_k      # svd_v_core = Vᵀ[:k,:] @ L⁻¹ (whitened path)
updated_uᵀ = lstsq(x', y).solution
```

### 5. fasterprune factors and relative error

```
append_u = updated_uᵀᵀ @ √Σ_k
append_v = √Σ_k @ svd_v_core
err(Ŵ) = ‖y − x@Ŵᵀ‖_F / ‖y‖_F     # error uses new_w, updated_error uses append_u@append_v
```

## Input constraints

- `weight` / `x` / `y` are 2-D float tensors with x (P, N) and y (P, M);
- on the whitened path `scaling_diag_matrix` must be an (N, N) 2-D float tensor; it must not be passed when `direct_update=True`;
- `ratio` is a finite number in (0, 1] (bools do not count).

Return `dict(k, svd_u, svd_v, new_w, updated_u_t, append_u, append_v, error, updated_error)`.

## Hints

- On the whitened path `svd_u @ (svd_v @ L)` is the rank-k optimal approximation of `W @ L` — the carrier of "truncate small singular values ⇔ low loss";
- after the closed-form update the error must not worsen (`updated_error ≤ error`);
- H accumulation and Cholesky live in upstream profiling; L arrives as an input.
