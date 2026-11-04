# BD-LoRA Implementation Notes

## Source and Structure

- Official PEFT `src/peft/tuners/lora/variants.py` (`BlockDiagonalLinear` + `BdLoraLinearVariant`, `LoraConfig.use_bdlora=BdLoraConfig`); paper arXiv:2510.23346 (NeurIPS 2025, Amazon); serving side vLLM PR #28136.
- `BlockDiagonalLinear.weight` has shape `(out_features, in_features // nblocks)` — nblocks shards stacked along dim=0; the forward is `einsum("bim,inm->bin")` with `first_dims` preserved (3-D input reshaped to (-1, in) and restored).
- `weight_as_blockdiagonal_matrix`: `torch.block_diag(*torch.chunk(weight, nblocks, dim=0))`; an A factor (r, in/nb) expands to (r, in), a B factor (out, r/nb) to (out, r).
- `BdLoraLinearVariant.init`: substring match `pattern in target_name`; both -> ValueError; `match_strict` with neither -> ValueError. lora_A uses `init_zero=False` (kaiming), lora_B `init_zero=True` (all-zero).
- merge: `_get_bdlora_delta_weight` first expands both factors via `weight_as_blockdiagonal_matrix`, then applies `transpose(weight_B @ weight_A, fan_in_fan_out) * scaling`; fp16/bf16 on CPU upcasts to float32.

## Judging Conventions and Quirks

- **The compact weight shape is the core account**: A is `(r, in//nb)` not `(r, in)`; B is `(out, r//nb)`. After expansion `bd_forward(x) == x @ bd.T`.
- kaiming fan_in reads dim1 of the weight BEFORE transposition = `in//nb` (the shard width), not the full in (same lesson as 808 LoRA-SB); a=√5; seed-anchored and reproducible.
- delta keeps the block-diagonal structure: off-diagonal < 1e-12; diagonal blocks = shard products `wb[i] @ wa[i]` (times scaling) — the full matrix multiply differs by ~2e-16 (accumulation order), property asserts use allclose.
- The merge identity `layer_forward == F.linear(x, base_w + delta, base_b)` is the strongest signal (m-lf-*/m-dl-* mutations are exposed through it or the anchors).
- The `fan_in_fan_out=True` branch: `(weight_a.t() @ weight_b.t()) * scaling == delta.t()` — the test checks this identity directly.
- classify error messages copy the official wording verbatim ("matches both A and B block-diagonal patterns" / "matches neither A nor B").
- The divisibility check lives in both `bd_forward` and `init_factors` (officially in `BlockDiagonalLinear.__init__`; moved to the two judging entry points).
- einsum("bim,inm->bin") differs from manual block `cat([x_i @ w_iᵀ])` by ~1e-16; anchors follow the einsum path.
