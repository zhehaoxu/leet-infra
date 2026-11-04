# Sparsity: SliceGPT Orthogonal Rotate-and-Slice Step (PCA Damping + Shortcut Chain Bookkeeping)

## Problem

SliceGPT neither zeroes weights nor quantizes them — it changes the basis of the whole hidden space: an orthogonal rotation built from the eigenvectors of the calibration signals' correlation matrix, then **directly slices away the trailing low-energy dimensions** (a global rank reduction d → d'), with shortcut rotation matrices chaining the separated blocks back together. Implement one-layer rotate-and-slice `solve(layer_inputs, mlp_ln_inputs, layer_outputs, w_qkv, w_o, w1, w2, attn_in_dim, attn_out_dim, mlp_out_dim, o_bias=None, ignore_masks=None)`:

```text
# pca (official pca_calc, once per explicit signal group)
H = Σ_batch Σ_token x xᵀ          # tokens with mask==0 zeroed first
damp = 0.01 · mean(diag(H)); add damp to H's diagonal   # anti-degeneracy
torch.linalg.eigh(H) → reorder columns by descending eigenvalue → Q (columns = principal components)

# 1. Q_in (layer_inputs): rotate attention inputs
W_qkv ← (W_qkv @ Q_in)[:, :attn_in]     # rotate in float64, cast back to float32
attn_shortcut = Q_in.T[:attn_in, :]     # float32

# 2. Q_mid (mlp_ln_inputs): rotate attention output / MLP input
attn_shortcut ← attn_shortcut @ Q_mid[:, :attn_out]
W_o ← (Q_mid.T @ W_o)[:attn_out, :];  b_o ← (Q_mid.T @ b_o)[:attn_out]
mlp_shortcut = Q_mid.T[:attn_out, :]    # float32
W_1 ← (W_1 @ Q_mid)[:, :attn_out]       # mlp input dim ≡ attn output dim

# 3. Q_out (layer_outputs): rotate MLP output
mlp_shortcut ← (mlp_shortcut @ Q_out)[:, :mlp_out]
W_2 ← (Q_out.T @ W_2)[:mlp_out, :]
```

Return `dict(w_qkv, w_o, o_bias, w1, w2, attn_shortcut_q, mlp_shortcut_q, q_in, q_mid, q_out, eig_vals)` — the three descending eigenvalue arrays are returned alongside the weights (the damping is diagonal-only and observable).

## Key Semantics

- **Slicing is rank, not weight values**: W is sliced by rows/columns in the rotated basis with the remainder kept as-is — a different family from magnitude/Wanda's elementwise zeroing or SparseGPT's per-column reconstruction.
- **Damping is diagonal-only**: `H[diag,diag] += damp`; off-diagonal untouched (adding to all of H changes the eigenvectors).
- **eigh ascending → descending reorder**: the descending argsort reorders both eigenvalues and column vectors; slicing takes `[:, :k]`, the **highest-energy** components.
- **Two-stage shortcut concatenation**: `attn_shortcut = Q_in.T[:attn_in] @ Q_mid[:, :attn_out]` and `mlp_shortcut = (Q_mid.T[:attn_out] @ Q_out)[:, :mlp_out]` — slice Q_in's rows / Q_mid's columns first, matmul in float32 (Q cast to dtype first).
- **Sequential-block dimension consistency**: the MLP input dimension always equals the attention output dimension (the official ForwardSlicingScheduler); one `attn_out_dim` governs both.
- **Functional**: clone before mask zeroing; the inputs (signals/weights/masks) are never modified.

## Judgement-domain Conventions

- The official code takes inter-block signals from real model forwards; the judgement domain takes `layer_inputs / mlp_ln_inputs / layer_outputs` as explicit list[(B,S,d)] groups (H accumulated per batch within each group).
- Slicing dimensions are explicit arguments (no scheduler); `final_orientation='pca'` (the random branch is out).
- Weight rotation happens in float64 then casts back to float32; shortcut concatenation uses float32 per the official code.
- Eigendecomposition convention: must use `torch.linalg.eigh` (same environment/backend, deterministic signs), with a non-degenerate test spectrum (adjacent eigenvalue gaps > 1e-3).
- The official code validates nothing; the judgement domain adds signal-group/mask-alignment-and-0-1/weight-shape/slice-dimension checks.

## Example

With d=8, attn_in=5, attn_out=6, mlp_out=7, two masked batches: the three PCAs each give an 8×8 orthogonal Q (descending, non-degenerate eigenvalues); W_qkv is sliced (8,8)→(8,5), W_o (6,8) (bias sliced to (6,)), W_1 (16,6), W_2 (7,16); attn_shortcut (5,6), mlp_shortcut (6,7). Tokens with mask 0 never enter H; Q_in satisfies `H @ q = q·diag(vals)` on layer_inputs.

Data scale: d ≤ 32, f ≤ 4d, up to 4 batch groups, float32 weights in [-1, 1].
