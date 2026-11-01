# Multimodal: FLUX Single-Stream Assembly and Shared vec Accounting

## Statement

FLUX's transformer has two stages: double blocks (two-stream MMDiT) followed by single blocks (single-stream DiT). This problem transcribes **all the wiring of the latter stage**: given the text and image streams after the double blocks, the two conditional embedding inputs, a weight dict, and the attention output (attention internals excluded), return every intermediate ledger of the single-stream stage plus the final image stream.

Inputs:
- `enc_out` [B, Lt, D]: the text stream after the double blocks;
- `img_out` [B, Li, D]: the image stream after the double blocks;
- `t_emb` [B, D] / `pooled_proj` [B, D]: the timestep embedding and pooled text projection;
- `weights`: `single_mod_w/b` [3D,D]/[3D], `proj_mlp_w/b` [M,D]/[M], `proj_out_w/b` [D,D+M]/[D];
- `attn_output` [B, Lt+Li, D]: the attention output over the concatenated stream (caller-provided).

Return the 5-tuple `(vec, shift_msa, scale_msa, gate_msa, hidden_final)`.

## Rules (each mirrors diffusers v0.32.1 verbatim)

1. **vec assembly**: `vec = t_emb + pooled_proj` — **elementwise addition**, not concat.
2. **Stream concat**: `hidden = cat([enc_out, img_out], dim=1)` — **text stream first**.
3. **3-parameter modulation**: `mod = single_mod(silu(vec))`, chunk(3) into shift_msa / scale_msa / gate_msa (the single block uses 3 parameters, not the double block's 6).
4. **Modulated norm**: `norm_x = layer_norm(hidden, eps=1e-6, no affine) * (1 + scale_msa) + shift_msa` ([:, None] broadcast).
5. **Shared input**: `norm_x` feeds both attention and MLP; `mlp = gelu_tanh(linear(norm_x))` — **linear first, GELU after** (official `act_mlp(proj_mlp(x))`).
6. **Channel concat**: `fused = cat([attn_output, mlp], dim=2)` (attn first), `out = gate_msa * proj_out(fused)`, `hidden = hidden + out` (residual over the whole concatenated stream).
7. **Slice back**: `hidden_final = hidden[:, Lt:, :]` — take the image stream by the text length.

## Example

```text
Input:  B=1, Lt=1, Li=1, D=6 (full case in selftest seed=4)
Output: vec [1,6]; shift/scale/gate each [1,6] (= thirds of linear(silu(vec)));
        hidden_final [1,1,6] (= img_out + gate * proj_out(cat([attn, mlp])))
(with the gate weights zeroed, hidden_final equals img_out elementwise)
```

## Constraints

- B=1..3, Lt/Li=1..8, D=4..16; all tensors float32/float64 sharing one dtype; non-finite values raise ValueError.
- Missing weight keys, weight shapes inconsistent with the derived D/M, or an attn_output shape other than [B, Lt+Li, D] raise ValueError.
- Non-contiguous inputs accepted; inputs are not modified; F.scaled_dot_product_attention is forbidden.

## Hints

- The two easiest-to-reverse points: vec is addition, not concat; the stream concat puts text first and the slice uses Lt.
- The MLP's activation placement: `gelu(linear(x))`, not `linear(gelu(x))`.
- The gate multiplies only proj_out's output, never the residual.
