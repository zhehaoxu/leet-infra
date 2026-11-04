# Notes: Basis Sharing Cross-Layer Shared-Basis Factorization

## Provenance

The Basis Sharing paper (arXiv:2410.03765, ICLR 2025, Jingcun Wang et al., TU Darmstadt + UNL, official repo `TUDa-HWAI/Basis_Sharing`). Motivation: when SVD compression runs per layer independently, the learned basis vectors across layers are highly correlated — sharing one basis across a layer group amortizes "a full basis per layer" into "one basis per group plus narrow per-layer coefficients". The paper systematically studies **which weight matrices are worth sharing** (q/k/v/o and gate/up/down each get their own basis pool and grouping) and **group selection**, beating whitened-SVD baselines such as SVD-LLM at large compression ratios. The calibration preprocessing reuses SVD-LLM's Cholesky whitening directly (calib.py credits it).

**Dedup boundaries**: single-layer "Cholesky whitening + truncated SVD" is already pinned by `sparsity-svdllm-whitened-truncation` (=773); this problem's increments are the **cross-layer joint concatenation factorization + shared basis + per-layer coefficient split + least-squares re-solve + same-input calibration reuse**. The KV-side cross-layer SVD share is `kv-xkv-layer-svd-share` (KV-cache projection dims, not weight matrices).

## Line-by-line translation of the official implementation

```python
# group.py: Group._init_basis_coefficient (llama2 path)
w = [model[layer].get_submodule(name).weight.data.T for layer in member]
w = torch.cat(w, -1).double()               # (in, m*out) wide matrix
w = s @ w                                   # SVD-LLM whitening (S = chol(H).T)
u, sigma, v = torch.svd(w)
self.basis = invs @ u @ torch.diag(sigma)   # (in, in) full; truncated in change_basis
self.coefficient = v.T                      # (in, m*out); column segments = layers

# change_basis / change_coefficient: truncate to num_basis
basis_name_weight = self.basis[:, :num_basis]      # (in, k)
weight = co[layer][name][:num_basis, :]            # (k, out_l)

# update_model: closed-form coefficient re-solve after the basis changes
u = basis[:, :]                                    # (in, k), already truncated
inv = torch.inverse(u.T @ xtx @ u)
vt = w.T @ xtx @ u @ inv                           # (m*out, k)
v = vt.T                                           # (k, m*out); column segments = layers
```

Three key readings:

1. **Whitening acts on the wide matrix**: `w = s @ w_cat` is a single multiply (numerically identical to per-layer multiplies followed by concat via the distributive law — recorded as an equivalent mutation, do not pin); \(S = \mathrm{chol}(H)^\top\) is upper triangular with \(H = \Sigma x^\top x\) the calibration Hessian.
2. **Two paths to the coefficients**: initial \(C = v^\top\) split into column segments (transpose of the SVD right factor, segments = layers); update \(C = (W^\top H B)(B^\top H B)^{-1}\) (computed as (m·out, k), then transposed, segments = layers). **With \(H = S^\top S\) the two coincide**: the H-weighted LS objective expands to exactly the optimal low-rank approximation in the whitened domain. The update stage exists because the basis keeps being trained (lora.py) or gets replaced.
3. **Same-input projections share calibration**: q/k/v all consume hidden_states and share H (q/v reuse k's calibration file); gate/up likewise. o/down are independent. The map appears in both `get_calib_data` and the basis_name branch of `update_model`, with consistent semantics.

**The forward chain** (llama.py): `y = q_proj(q_basis(x))`, i.e. \(x \to B \to C_l\); equivalent to an ordinary linear layer with the reconstructed \(\hat W_l = (BC_l)^\top\). On the MLP side `down(act(gate_basis(x)·coef) * up_basis(x)·coef)` — three projections each with their own shared basis.

## Pitfalls

1. **Concat axis and transpose**: llama weights transpose first, then concat along -1 (column segments = layers); skipping the transpose or using axis=0 breaks the "segment = layer" premise for coefficient splitting (m-notrans / m-cat-axis).
2. **Three parts of the shared basis**: invs (de-whitening), diag(σ) (energy weighting), and the top-k columns (truncation) are all required (m-basis-noinv / m-basis-nosigma); taking the bottom k columns breaks the energy ordering (m-basis-bottomk).
3. **Coefficient segment offsets**: the segment start is \(i \cdot out\), not \(i\) (m-coef-offset); truncation takes the first k **rows** (rows of vᵀ = component order), taking columns is a transpose mix-up (m-coef-rowswap); dropping the truncation breaks the parameter account (m-coef-nok).
4. **The LS closed-form structure**: the Gram inverse is \((B^\top H B)^{-1}\), not \((B^\top B)^{-1}\) (m-ls-noinv); H weights at \(W^\top H B\) (m-ls-noH); transposition happens after solving (m-ls-rowsplit).
5. **Reconstruction/forward factor order**: \(\hat W = (BC)^\top\), not \(CB\) (m-recon-order); a wrong order in \(x\,B\,C_l\) breaks both shape and values (m-fwd-order).
6. **The calibration map**: the two reuse rules q/v→k and gate→up (m-calib-qv / m-calib-gate).
7. **Equivalent-mutation record**: whitening "per layer then concat" vs "concat then whiten" is identical by the distributive law (m-white-perlayer SURVIVED, do not pin again).

## Cross-check account

Anchors dumped from the sandbox (numpy float64, seed 57): in=8/out=6/m=3/k=4, H built as \(X^\top X\) (SPD), S = chol(H)ᵀ upper triangular, thin SVD factors as explicit inputs (no decomposition at solve time, signs frozen). Four cross-check lines — (1) full-rank reconstruction: \(B_{full} C_{full} = W_{cat}\) (exact under invertible invs); (2) the full-rank LS solution equals the full \(v^\top\); (3) **the two-stage identity**: under the SVD basis the LS re-solve equals the initial coefficients (numeric verification of \(H = S^\top S\)); (4) on a perturbed basis LS satisfies the normal equations (residual ≈ 0) and differs observably from the initial coefficients. Forward ≡ \(x\hat W^\top\) same-source check. Of 19 mutants 18 CAUGHT plus 1 distributive-law equivalent.

## Complexity

Factorization costs \(O(in^2 \cdot m \cdot out)\) (offline, once); compressed parameters \(in \cdot k + m \cdot k \cdot out\), saving \((m{-}1)(in - k) \cdot out\) for an m-layer group; forward per layer \(O(BN(in\cdot k + k\cdot out))\), a \(k / \min(in,out)\) factor below the original \(O(BN \cdot in \cdot out)\). The basis-storage amortization grows linearly with group depth.
