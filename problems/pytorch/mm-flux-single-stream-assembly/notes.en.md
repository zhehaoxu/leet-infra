# Notes: FLUX Single-Stream Assembly and Shared vec

## Approach

This problem transcribes the single-stream stage of diffusers `FluxTransformer2DModel`: everything from after the double blocks to the output slice. The solution follows the official order in five steps: vec assembly (`t_emb + pooled_proj`, elementwise addition), `cat([enc, img], dim=1)` stream concat (text first), AdaLayerNormZeroSingle's 3-parameter modulation, the shared normalised input feeding both attention and MLP plus channel-wise concat, then gate multiplication + residual + slicing by text length. There is no scalar arithmetic at all, so verification focuses entirely on wiring direction and order.

## Deliberate deviations from the official code

- **Batch dim**: the official code supports any batch; this problem bounds B=1..3 with consistency checks.
- **fp16 clip**: the official `FluxSingleTransformerBlock` clips to (-65504, 65504) under fp16; this problem only accepts float32/float64, where the clip is a no-op — recorded here, invalid dtypes raise ValueError.
- **Multiple blocks**: the official model runs 38 single blocks all sharing the same `temb` (vec). This problem runs one block but returns the three modulation tensors as ledgers, enough to pin "shared vec, per-block modulation"; more blocks are just repeated application with no new accounting.
- **ControlNet residual**: the official single blocks add ControlNet samples only to `hidden[:, encoder_len:]` (the image part); out of scope here, left as a candidate.
- **Attention internals** (RMSNorm on qk, RoPE, `pre_only=True` with no output projection) are excluded; `attn_output` is caller-provided, matching the boundary style of `mm-flux-joint-block-modulation` (order 649).

## Five pinned accounting points

1. **vec is addition, not concat**. Official `conditioning = timesteps_emb + pooled_projections` — the timestep embedding and the pooled text projection are added elementwise (both [B, D]); the intuitive mistake is to concatenate.
2. **Text stream comes first**. Official `torch.cat([encoder_hidden_states, hidden_states], dim=1)`, paired with the final slice `hidden[:, encoder_len:]`; reversing the order corrupts the LayerNorm statistics and the gate target.
3. **The single block uses 3-parameter modulation**. AdaLayerNormZeroSingle's linear outputs `3*emb`, chunk(3) into shift_msa / scale_msa / gate_msa — unlike the double block's (order 649) 6 parameters. The normalisation form is `norm(x) * (1 + scale) + shift`.
4. **norm_x feeds attention and MLP from one source**. The official code calls `self.norm(hidden_states, emb=temb)` once, and the same `norm_hidden_states` enters both `self.attn` and `self.proj_mlp`; feeding the modulated normalised input to the MLP is the boundary against "MLP eats the raw hidden".
5. **attn and mlp concat along channels into proj_out**. Official `torch.cat([attn_output, mlp_hidden_states], dim=2)` (attn first), `proj_out` input dim `dim + mlp_hidden_dim`, output `gate * proj_out(fused)` plus residual.

## Hand-computation traps (located via cross-checking)

- **GELU placement**: official `self.act_mlp(self.proj_mlp(norm_hidden_states))` is **gelu(linear(x))** — linear first, GELU after. The oracle's first draft wrote `linear(gelu(x))`, which broke every block output; intermediate-by-intermediate comparison against the solution located it. This is exactly the divergence between "activation on the proj_mlp output" and "activate the normalised input before projecting"; the `gelu-before-linear` mutation pins it.
- **Zeroing ≠ clearing**: setting the gate chunk of `single_mod_b` to a large negative value does not zero the gate — the weight term `W·silu(vec)` can push it back; the whole W and b segments must be zeroed together. `check_gate_residual` / `check_scale_one_plus` are built that way.
- **Slicing contiguity**: `big[:, :2]` on the leading dim is still contiguous; to build a non-contiguous view, transpose first then slice (`big.transpose(1,2)[:, :2]`).

## Independent verification

The oracle uses no torch linear algebra: `_linear_row` does matmul with scalar double loops, `_silu` / `_gelu_tanh` / `_layer_norm_row` (biased variance, eps=1e-6) are hand-written elementwise, the stream is assembled per token in lists, then converted back to torch for comparison (allclose 1e-5 + dtype equality). The 12 `check_*` cover: the additive vec semantics (including `torch.equal(vec, te+pp)`), concat order (swapping enc/img must change the result), the 3-chunk boundaries, the shared MLP input (substituting norm_x for attn_output and recomputing), the (1+scale) degenerate case, gate-zero degenerating to residual, slicing by Lt, float64, non-contiguous views, input immutability, and nine groups of invalid inputs.

All 12 observable mutations CAUGHT (`vec-sub`, `img-first-concat`, `no-one-plus-scale`, `mlp-raw-input`, `gelu-before-linear`, `mlp-first-fused`, `gate-broadcast-skip`, `no-residual`, `slice-from-zero`, `eps-1e-5`, `silu-to-gelu-mod`, `six-chunk`). One lesson recorded: the first-draft `vec-concat` mutation was "concat then split and add back", mathematically equivalent to the addition and hence an invalid mutation; replacing it with subtraction got it caught immediately.

## Complexity

Let L = Lt + Li, M = mlp_hidden_dim. Per block: modulation O(D²), MLP O(L·D·M), proj_out O(L·D·(D+M)); total O(L·D·(D+M)); besides the output and three [B,D] ledgers no extra intermediates are kept.

## Source and scope

Based on diffusers v0.32.1 sources (GitHub API contents + base64): `transformer_flux.py::FluxSingleTransformerBlock.forward` (residual / modulation / concat / gate / clip), `transformer_flux.py::FluxTransformer2DModel.forward` (the `cat([encoder, hidden])` concat and the `hidden[:, encoder_hidden_states.shape[1]:]` slice), `normalization.py::AdaLayerNormZeroSingle` (3-chunk and the `(1+scale)` form), `embeddings.py::CombinedTimestepTextProjEmbeddings` (`conditioning = timesteps_emb + pooled_projections`). The vec upstream (Timesteps sinusoidal encoding ×1000, PixArtAlphaTextProjection two-layer projection) is excluded — pure DSP/MLP; the accounting point is the assembly semantics. Boundary with `mm-flux-joint-block-modulation` (order 649, double-block six parameters): this problem is the single-stream assembly and 3-parameter shared accounting.

```text
.venv-eval/bin/python scripts/add_problem.py --selftest mm-flux-single-stream-assembly --mode submit
```
