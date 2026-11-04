# VeLoRA Implementation Notes

## Source and Structure

- Official repo `roymiles/VeLoRA` (Apache-2.0), core file `velora/velora.py`; NeurIPS 2024, arXiv:2405.17991 (Huawei Noah's Ark Lab).
- `VeLoRAWrapper.forward`: cast x to fp16 -> `init_embed_` (first-batch gate) -> `zp = einsum('h s, r s -> h', x, embed)`; rank=1 gives output (H,).
- `VeLoRA(torch.autograd.Function)`: forward saves (zp, weight, v); backward rebuilds `input_hat = einsum('h, r s -> h s', zp, v).view(ctx.input_shape)`, then `grad_input = grad_output @ weight`, `grad_weight = transpose(1,2) @ input_hat`, `grad_bias = grad_output.sum(1)`, and the returned grad_weight is multiplied by `ctx.scale`.
- `functional_velora_linear_fn`: zp projects the **post-dropout lora input** (identity dropout in the judging domain, i.e. x itself); `VeLoRA.apply(lora_result, lora_A, None, zp, embed, velora_scale)` passes the scale into backward to multiply ONLY lora_A's gradient; the B side goes through plain autograd.

## Judging Conventions and Quirks

- The official backward returns the per-sample (B,M,D) term **without summing** — real autograd sums the leading dim when consuming it; the judging domain takes the equivalent account `sum(0)` (stated in both the description and the tests).
- `init_embed_`'s `batch_average_once` carries `assert rank == 1`; mean(0) then `F.normalize(dim=-1)`; the initted gate ensures it runs once.
- The rebuild is lossy but the residual is orthogonal to embed: hat·v = zp·|v|² = zp = x·v ⇒ (hat−x)·v = 0; round trip `project(reconstruct(zp)) == zp`.
- **The asymmetry is official semantics**: grad_a flows through the reconstructed activations, grad_b through the real `a_out`; swapping grad_b to the reconstructed path is a catchable mutation (m-lb-ares).
- `velora_scale` multiplies only lora_A's gradient; the forward result, grad_input and grad_b never see it.
- einsum and mul-sum use different accumulation orders (allclose rather than equal in float); all anchors are float64 with 1e-9 tolerance.
- The reshape row order is (B, N, G): a sub-token group-order mutation (transpose(1,2)) changes every zp — m-pr-order.
- grad_bias sums over the **sequence dim** dim=1 (B,L,M -> B,M); summing over batch (sum(0)) is a catchable mutation.
