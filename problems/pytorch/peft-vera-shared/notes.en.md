# PEFT: VeRA Shared Frozen Base Matrices with Learnable Scaling — Algorithm Notes

## 1. Provenance and core idea

Kopiczko et al. (ICLR 2024, arXiv:2310.11454). Of LoRA's \(r(d_\text{out}+d_\text{in})\) parameters, the "learned directions" are highly redundant — VeRA turns \(A,B\) into a **cross-layer shared random frozen base** (generated once, reused by all layers), with each layer learning only two scaling vectors:

- \(\lambda_d\in\mathbb R^r\): acts on the low-rank **hidden** dimension (scaling the r components of \(xA^\top\));
- \(\lambda_b\in\mathbb R^{d_\text{out}}\): acts on the **output** dimension (scaling each component after projecting back to d_out).

It is equivalent to a "per-layer diagonal reweighting" of the shared base. On LLaMA/GPT-2 the paper reaches fine-tuning quality close to LoRA with \(d_\text{out}+r\) parameters per layer (about one percent of LoRA's count).

## 2. Two equivalent forward paths (judged points)

**Functional** (PEFT's forward as written):

\[y=xW_0^{\!\top}+\lambda_b\odot\Bigl(\bigl(\lambda_d\odot(xA^{\!\top})\bigr)B^{\!\top}\Bigr)\]

**Matrix form** (after merge):

\[\Delta W=(\text{diag}(\lambda_b)\,B)\,(\text{diag}(\lambda_d)\,A),\qquad y=x\,(W_0+\Delta W)^{\!\top}\]

The `check_matches_oracle` oracle takes the matrix form and the reference solution the functional form, pinning the equivalence by cross-check (the shape this collection's red line "only assert rebuilt ΔW / merged weights" takes on VeRA: however you compute it, \(\Delta W\) must be the same). Note \(\lambda_d\) multiplies the **columns** of \((N,r)\) and \(\lambda_b\) the **columns** of \((N,d_\text{out})\) — do not flip either broadcast direction.

## 3. Initialization and parameter account

- PEFT's `reset_vera_parameters`: \(\lambda_b\) all zeros, \(\lambda_d=d_\text{initial}=0.1\). At the start \(\Delta W=0\) and the adapter "does nothing" — `check_zero_lambda_b_identity` verifies this.
- Each layer has \(d_\text{out}+r\) trainable parameters versus LoRA's \(r(d_\text{out}+d_\text{in})\): `check_param_account` asserts this on random shapes.
- `check_lambda_b_per_output`: doubling one entry of \(\lambda_b\) changes only the corresponding **output column**, and by exactly that column's original adapter amount — the per-output gating semantics.

## 4. Division of labor within the collection

- `peft-lora-forward`/`peft-lora-merge`: LoRA with per-layer independent A/B.
- `peft-dora`: magnitude-direction decomposition (per-layer independent factors + magnitude vector).
- This problem: the extreme parameter-saving form of a cross-layer shared base + per-layer scaling vectors.

## 5. Verification approach

`check_example` (\(d_\text{in}=2,d_\text{out}=2,r=1\)): \(A=[1,1]\), \(B=[1,-1]^{\!\top}\), \(\lambda_d=2\), \(\lambda_b=[1,-1]\), \(x=[1,2]\) → hidden \(=3\cdot2=6\), adapter \([6,-6]\), times \(\lambda_b\) gives \([6,6]\), plus base \([1,2]\) → \(y=[7,8]\), \(n=3\); `check_matches_oracle` cross-checks the two paths (float64 internally, float32 out, tolerance \(10^{-5}\)).
