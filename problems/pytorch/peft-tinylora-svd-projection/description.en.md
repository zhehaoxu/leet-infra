# PEFT: TinyLoRA Frozen SVD Factors and Random Projection Composition

## Problem

**TinyLoRA** (Learning to Reason in 13 Parameters, arXiv:2602.04118, Meta) pushes the LoRA-XS skeleton of "frozen SVD factors + a small middle matrix R" to the extreme: R is no longer trained independently but written as a **linear combination of fixed random projection matrices** \(R=\sum_i v_i P_i\) — the only trainable parameter is a length-\(u\) vector \(v\), as small as 13 bf16 values in the paper, enough to take Qwen2.5-8B to 91% on GSM8K. The official reference implementation is `ANRGUSC/TinyLoRA`, and it entered HF PEFT via PR #3024 (`tuners/tinylora`).

This problem replicates the deterministic accounts of PEFT `TinyLoraLayer`:

- **SVD split** (`_init_svd`): \(W=U\,\mathrm{diag}(S)\,V^\top\) with \(\sqrt{S}\) **split equally to both sides** — \(A=\mathrm{diag}(\sqrt{S_{:r}})\,V^\top_{:r,:}\ (r,d_\text{in})\), \(B=U_{:,:r}\,\mathrm{diag}(\sqrt{S_{:r}})\ (d_\text{out},r)\); the rank is clamped by the short side, \(r_\text{act}=\min(r,\min(d_\text{out},d_\text{in}))\). The equal split balances the norms of A and B.
- **Projection init** (`_init_projection`): \(P\in\mathbb{R}^{u\times r\times r}\sim\mathcal{N}(0,1/r)\) (std \(=1/\sqrt r\), the Johnson–Lindenstrauss convention), drawn from `torch.Generator().manual_seed(base_seed + layer_idx)` per layer; P is a frozen buffer, not a parameter.
- **R composition** (`_compute_R`): `einsum("i,ijk->jk", v, P)`; v is zero-initialized ⇒ \(\Delta W=0\) at the start of training.
- **Merge/forward**: \(\Delta W = B@R@A\); the forward delta is \(x\,A^\top R^\top B^\top\) (the LoRA-XS chain of three linears: A→R→B); the two are mathematically equivalent.
- **weight_tying groups** (model.py): target layers get deterministic indices in injection order; `num_groups = max(1, round(n·(1−wt)))` (wt=0 no sharing, wt=1 full sharing), `group_size = max(1, n//num_groups)`, `group_idx = min(idx//group_size, num_groups−1)`; layers in the same group share one v — trainable parameters = num_groups×u.

## Rules

- `mode="svd_split"`: inputs U/S/Vh (full SVD factors, computed offline) and r; returns \((A,B,r_\text{act})\).
- `mode="init_projection"`: inputs u, r, seed; returns P \((u,r,r)\), bit-identical on replay with the same seed.
- `mode="compute_R"`: inputs v and P; returns \(R\ (r,r)\).
- `mode="delta_weight"` / `mode="forward_delta"`: inputs B, R, A (and x).
- `mode="tie_groups"`: inputs n and wt; returns `{"num_groups", "group_size", "group_indices"}`.
- Everything in float64; dropout=0.

## Example

\(d_\text{out}=6, d_\text{in}=8, r=4, u=6, seed=7\):

```
svd_split(U, S, Vh, 4) -> A:(4,8), B:(6,4), actual_r=4
init_projection(6, 4, 7) -> P:(6,4,4)
compute_R(v, P) -> R:(4,4)
delta_weight(B, R, A) -> dW:(6,8), and x @ dW.T == x@Aᵀ@Rᵀ@Bᵀ
tie_groups(6, 0.5) -> {num_groups: 3, group_size: 2, group_indices: [0,0,1,1,2,2]}
```

When \(r=8\) exceeds the short side (e.g. a \(5\times4\) weight), \(r_\text{act}\) is clamped to 4.

## Constraints

- The SVD factors are all explicit inputs (the judge never runs an SVD).
- Column norm of \(B\) = row norm of \(A\) = \(\sqrt{S_i}\) (a direct corollary of the equal split, spot-checked in the tests).
- The grouping ledger satisfies: number of groups = number of distinct group_idx values, trainable parameters = groups×u.
