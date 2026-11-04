# PEFT: VeLoRA Rank-1 Sub-Token Activation Compression

## Problem Description

**VeLoRA** (NeurIPS 2024, arXiv:2405.17991, official `roymiles/VeLoRA`, `velora/velora.py`) compresses the intermediate activations that backpropagation must keep in memory onto a **rank-1 subspace**: the forward pass splits every token along the feature dim into `num_groups` sub-tokens and projects each onto one **fixed unit vector** (rank=1 embed), storing a single scalar `zp` per sub-token; the backward pass **coarsely rebuilds** the activations from the same vector. The projection vector is initialized once from the first batch's statistics (`batch_average_once`) and then frozen. Key asymmetry: **only lora_A's gradient flows through the reconstructed activations**, while lora_B's gradient uses the real lora_A outputs.

This problem replicates the deterministic account of the official implementation:

- **Sub-token projection**: x (B,N,C) -> reshape (B·N·G, C/G) -> `zp = einsum('h s, r s -> h', flat, embed)`; embed is an L2-normalized unit row vector.
- **batch_average_once init**: `embed = normalize(data.mean(0), p=2, dim=-1)` — first-batch mean, L2-normalized, then frozen.
- **Backward reconstruction**: `input_hat = einsum('h, r s -> h s', zp, v).view(input_shape)` — a lossy rank-1 coarse rebuild.
- **Gradient account**: `grad_input = grad_output @ weight`; `grad_weight = velora_scale · Σ_b grad_output[b]^T @ input_hat[b]` (the official backward returns the per-sample (B,M,D) term whose leading batch dim is summed by autograd — the equivalent account is sum(0)); `grad_bias = grad_output.sum(1)` (over the sequence dim, returned only when a bias exists).
- **LoRA block**: `velora_scale` multiplies grad_a ONLY; grad_b uses the **real** `a_out = x @ lora_a.T` times the lora `scaling`; the forward is the plain LoRA path `base + (a_out @ lora_b.T)·scaling`; `grad_input = grad_out_m @ weight + scaling·((grad_out_m @ lora_b) @ lora_a)`.

## Task

- `mode="project"`: inputs x, embed, num_groups; return zp (B·N·G,).
- `mode="init_embed"`: input data (M,S); return the unit vector embed (S,).
- `mode="reconstruct"`: inputs zp, v (1,S), shape; return the rebuilt activation with the given shape.
- `mode="grads"`: inputs input_hat, weight, grad_output, velora_scale, use_bias; return dict(grad_input, grad_weight, grad_bias); grad_bias=None when there is no bias.
- `mode="lora_block"`: inputs x, weight, bias, lora_a, lora_b, scaling, num_groups, embed, velora_scale, grad_out_m, grad_out_r; return dict(result, grad_a, grad_b, grad_input).

## Example

seeds 0/2/3/4, float64:

```
project(x (2,3,8), embed (1,2), num_groups=4) -> zp (24,)
  zp[0] = -2.3036448720441434, zp[-1] = -1.3527679700105146
init_embed(data (7,5)) -> embed (5,), embed[0] = -0.7252327797613423, norm = 1
reconstruct(zp, v, (2,4,8)) -> hat (2,4,8), hat[0,0,0] = -0.6753214360032626
grads(input_hat (2,5,6), weight (4,6), grad_output (2,5,4), velora_scale=0.1)
  -> grad_weight[0,0] = 0.22305672787004252, grad_input[0,0,0] = -0.0783169401206513
lora_block(...) -> grad_a[0,0] = -0.3717755681072157, grad_b[0,0] = 1.490616682369623
```

## Constraints

- embed is an L2-normalized unit row vector; the official `batch_average_once` supports rank=1 only.
- The reshape row order is (B, N, G) — a wrong sub-token group order changes every zp; C is divisible by num_groups.
- `velora_scale` applies to grad_weight (grad_a in the lora block) only; it never touches grad_input or the forward output.
- The reconstruction is lossy: `hat ≠ x`, but the residual is orthogonal to embed inside every sub-token chunk (round trip: `project(reconstruct(zp)) == zp`).
