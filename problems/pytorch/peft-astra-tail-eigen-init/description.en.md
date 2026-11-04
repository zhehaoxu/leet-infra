# PEFT: Astra Activation-Space Tail-Eigenvector Initialization

## Problem

**Astra** (arXiv:2602.19111, merged into HF PEFT as PR #3725, `init_lora_weights="astra"`) lets the **output-activation covariance** decide the LoRA initialization direction: a calibration forward collects each layer's output covariance, one **eigendecomposition** (`eigh`, ascending) follows, and the base weight is projected onto the eigenvector subspace — the trainable adapter carries the \(V\)-subspace component while the frozen base absorbs the residual. At initialization the model reproduces its original output **exactly** (zero delta), and training acts on the task-relevant subspace. The official README stresses: one eigendecomposition, **no weight SVD, no covariance inverse** (the boundary with CorDA).

This problem re-implements the deterministic account of the official `tuners/lora/astra.py` + `layer.py` `astra_init`:

- **Covariance accumulation** (hook account): each forward call flattens activations to \((N, dim)\), casts float, normalizes by the CALL's global \(|max|\) (`torch.max(out).abs()` — the **signed** max first, which differs from `abs().max()` when negatives dominate!), accumulates \(cov \mathrel{+}= Y^\top Y\) with a per-call count, then divides by the count.
- **Eigendecomposition**: `S, V = eigh(cov.float())` — **ascending** (\(S[0]\) smallest); \(S\) is kept full length, \(V\) is cropped to `[:, -r:]` (i.e. the top-\(r\) eigenspace under the ascending order).
- **astra_init**: \(A = V^\top W\) (\(r, in\)), \(B = V\) (\(out, r\)); \(\Delta = scaling\cdot B@A\), \(W \leftarrow W-\Delta\). \(B@A = V(V^\top W)\) is exactly the projection of \(W\) onto the \(V\) subspace.
- **Validations**: dtype in {float32, float16, bfloat16} (else TypeError); \(r \le out\_features\), \(V\) shape \((out\_dim, r)\), NaN/Inf (all ValueError).

## Task

- `mode="cov_accumulate"`: batches (a list of activation tensors) -> (cov, sample_count); raises ValueError on NaN/Inf.
- `mode="eigens"`: cov, r, out_dim -> (S full length, V cropped); raises ValueError for invalid r.
- `mode="init"`: weight, v_r, scaling -> (lora_A, lora_B, W_new); validations as above.
- `mode="forward"`: x, lora_A, lora_B, W_new, scaling -> \(scaling\cdot x\,A^\top B^\top + x\,W_{new}^\top\).
- The covariance pipeline runs in the official float32 (`.float()`).

## Example

\(out\_dim=6, in\_dim=5, r=2\), W = randn(6,5), X = randn(32,6) (seed 2, float32 pipeline):

```
cov_accumulate([X]) -> cov (6,6), sample_count = 1
eigens(cov, 2, out_dim=6) -> S full (6,), S[0]=1.806457757949829, S[-1]=8.876955032348633
                            V_r (6,2), V_r[0,0]=0.7613240480422974
init(W, V_r, scaling=2) -> A[0,0]=-0.8271405696868896, W_new[0,0]=0.5683993101119995
                           W_new + 2*B@A == W (float32 domain)
forward(x, A, B, W_new, scaling=3) == x @ W^T (zero delta)
```

## Constraints

- `1 <= r <= out_features`; invalid r raises ValueError.
- The normalization divisor is `torch.max(out).abs()` (the absolute value of the signed max), not `out.abs().max()`.
- The covariance returns float32; A/B/W_new return the weight's original dtype.
