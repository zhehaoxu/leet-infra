# FLUX Double-Block Modulation Ledger: AdaLN-Zero Six-Parameter and Gated Residual Accounting

## Problem

FLUX's transformer consists of double blocks (image and text streams, each with its own AdaLN-Zero modulation, plus joint attention) and single blocks (concatenated streams, shared modulation). The block skeleton is a precise modulate-gate-residual protocol: `silu(temb)` through a per-stream Linear yields 6 parameters that modulate the LayerNorm output, gate the attention and FF branches, and add back the **original input**. Attention internals (QKV+softmax+RoPE) are covered elsewhere; this problem takes attention outputs as explicit inputs and tests the protocol ledger alone.

## Rules

Implement `solve(hidden_states, encoder_hidden_states, temb, weights, attn_img, attn_txt)` (batch=1), transcribing `FluxTransformerBlock.forward` stream by stream (independent `img_*` / `txt_*` weights), returning:

| Field | Shape | Meaning |
|---|---|---|
| `hidden_states` | `[1, Li, D]` | image-stream block output |
| `encoder_hidden_states` | `[1, Lt, D]` | text-stream block output |
| `img_mod` / `txt_mod` | `[6, D]` | six parameters, chunk order `[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]` |
| `img_norm_input` / `txt_norm_input` | `[Li, D]` / `[Lt, D]` | modulated values fed to attention |

The five steps per stream (upstream, verbatim):

1. `emb = Linear(silu(temb))`, `chunk(6)` into six parameters
2. `x_mod = LayerNorm(x, eps=1e-6, no affine) * (1 + scale_msa) + shift_msa` (goes to attention)
3. `x = x + gate_msa * attn` (**residual adds to the original input, not the modulated value**)
4. `h = LayerNorm(x) * (1 + scale_mlp) + shift_mlp`
5. `x = x + gate_mlp * ff2(GELU_tanh(ff1(h)))`

Validation: `ValueError` for batch ≠ 1 or non-tensor inputs.

## Example

```
gate_msa = gate_mlp = 0 (zero the corresponding weight rows and biases) → block output = input (residual identity)
One wrong entry in the six-parameter chunk order → both the img_mod ledger and outputs drift
```

## Constraints

- Small scale (`D=4`), judged in float64; the oracle is a pure-Python elementwise rewrite (SiLU/GELU-tanh/LayerNorm/Linear)
- The observable face of eps is constructed with a near-constant row (variance ~1e-6)
