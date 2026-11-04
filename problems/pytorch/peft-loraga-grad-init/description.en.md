# PEFT: LoRA-GA's Gradient-Approximation Initialization

## Problem Description

**LoRA-GA** (arXiv:2407.05000, Outsider565/LoRA-GA) aligns LoRA's **first-step gradient** with
full fine-tuning: it takes a **randomized low-rank SVD** of the small-sample accumulated
gradient, assigns A/B directions from the top-\(2r\) singular-vector bank in four modes,
rescales the factors in four modes, and finally **absorbs the adapter's initial output into
the base weight** (\(W \leftarrow W-\eta\,BA\)) — training starts exactly at the pretrained
model while the first-step update direction matches full fine-tuning, giving a 2-4x
convergence speedup.

This problem replicates the deterministic accounts of the official implementation
(custom peft fork, `tuners/lora/layer.py`):

- **grad_svd**: `svd_lowrank(grad.float(), q=min(4r, min(shape)), niter=4)`, \(V^\top=V.\mathrm{T}\);
  returns \(U\ (d_\text{out},q)\), \(S\ (q,)\), \(V^\top\ (q,d_\text{in})\), promoted to float64.
- **direction (4 modes)** (mutually exclusive slices of the bank):
  - `ArBr`: \(B=U[:,\,0{:}2r{:}2]\) (**even columns**), \(A=V^\top[1{:}2r{:}2,\,:]\) (**odd rows**) —
    interleaved picking;
  - `A2rBr`: \(B=U[:,:r]\), \(A=V^\top[r{:}2r,\,:]\);
  - `ArB2r`: \(B=U[:,r{:}2r]\), \(A=V^\top[:r,\,:]\);
  - `random`: Python `random.sample(range(2r), 2r)`, **indexes_A (rows) drawn first, then
    indexes_B (columns)**.
- **scale (4 modes)**: `gd` (\(A\mathrel{/}=\eta,\ B\mathrel{/}=\eta\)); `unit` (A/B orthogonal,
  untouched); `stable` (\(B,A \mathrel{*}= m^{0.25}/\sqrt\gamma\), \(m=d_\text{out}=B.\mathrm{size}(0)\));
  `weightS` (\(S(w)\mathrel{/}=\eta\), \(avg_s=\sqrt{S[:r]}.\mathrm{mean}()\), \(B,A \mathrel{*}= avg_s\) —
  the weight singular values are provided by the caller).
- **offset absorption**: \(offset = B@A\cdot\eta\), \(W' = W - offset\); with `norm_clip`,
  \(ratio=\max|W|/\max|offset|\); when \(ratio<1\), \(offset \mathrel{*}= ratio\) and
  \(A,B \mathrel{*}= \sqrt{ratio}\) — the **invariant** \((B\sqrt{ratio})@(A\sqrt{ratio})\cdot\eta =
  offset\cdot ratio\) always holds.

## Task

- `mode="grad_svd"`: grad (out,in), r, niter, seed in, return (U, S, Vt).
- `mode="direction"`: U, Vt, r, direction (and the random seed) in, return (B, A).
- `mode="scale"`: B, A, scaling, scale (plus stable_gamma / weight_singulars) in, return the
  rescaled (B, A) WITHOUT in-place mutation.
- `mode="offset_apply"`: B, A, scaling, weight, norm_clip in, return (new_weight, a_w, b_w).
- All floats in/out as float64; the SVD pipeline follows the official float32 internally.

## Example

\(d_\text{out}=6, d_\text{in}=8, r=2\), grad = randn(6,8) (seed 0), \(\eta=4\):

```
grad_svd(grad, 2, niter=4, seed=42) -> U:(6,6), S:(6,), Vt:(6,8)
  S = [4.716037750244141, 3.28708815574646, 3.007770538330078,
       1.8632816076278687, 1.1207000017166138, 0.9001481533050537] (bit-anchored float32 pipeline)
direction(U, Vt, 2, "ArBr") -> B:(6,2)=U[:, [0,2]], A:(2,8)=Vt[[1,3], :]
scale(B, A, 4.0, "gd") -> B/4, A/4
offset_apply(B, A, 4.0, W, norm_clip=True) -> (W - B@A*4, A, B) (shrunk by the invariant when ratio<1)
```

## Constraints

- `svd_lowrank` uses the official q=min(4r, min(shape)) and niter arguments; r<=0 raises
  ValueError.
- Invalid direction / scale enum values raise ValueError.
- The weight singular values for weightS are explicit inputs (no weight SVD rerun in the
  judging domain).
