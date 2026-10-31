# Notes: FLUX Double-Block Modulation Ledger

## Provenance

diffusers `models/transformers/transformer_flux.py` — `FluxTransformerBlock.forward` (fetched via the GitHub contents API on 2026-10-31, transcribed sentence by sentence); `AdaLayerNormZero` lives in `models/normalization.py` (`silu → Linear(dim, 6*dim) → chunk(6)`, affine-free LayerNorm eps=1e-6). FeedForward `activation_fn="gelu-approximate"` = the tanh GELU approximation.

## Key points and traps

1. **The residual adds back the original input** is the biggest trap: `hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output` — the pre-block `hidden_states`, not the modulated `norm_hidden_states` (that is only attention's input). The first pure-Python oracle added the residual to the modulated value; the torch intermediate-state comparison exposed it (oracle wrong, solution right).
2. **The observable face of the msa branch**: `shift_msa/scale_msa` only affect attention's input; with attention outputs as explicit inputs and no ledger entry for them, both parameters are completely unverifiable (mutation `no-plus-scale` survived for exactly this reason). Adding `img_norm_input`/`txt_norm_input` to the ledger made every mutation catchable — the general lesson: a problem that factors attention out must ledger the attention input too.
3. **Six-parameter chunk order**: `[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]`; the `img_mod` ledger rows follow the chunk order. A misalignment corrupts both the outputs and the ledger.
4. **Zero gates = identity**: zeroing both the weight rows and the bias segments of gate_msa/gate_mlp returns the output bitwise to the input (residual identity) — zeroing biases alone is ineffective (the weight contribution remains); the first-draft case tripped on this.
5. **The observable face of eps**: at D=4 with O(1) values, the LN eps 1e-6 vs 1e-5 difference is ~1e-6 — inside the 1e-5 tolerance. A near-constant row (variance ~1e-6, eps dominating the denominator) amplifies it to percent level.
6. **Two independent streams**: both streams share temb but have independent modulation weights; perturbing image-stream weights leaves text-stream output untouched (an `atol=1e-9` isolation assertion).

## Mutation testing (9/9 caught)

drop `1+scale`, drop SiLU, residual onto the modulated value, drop the FF gate, GELU exact, eps 1e-5, chunk misorder, FF without activation, shared modulation weights, batch validation removed. No equivalent mutation.

## Complexity

D=4, Li=2, Lt=3, microseconds.
