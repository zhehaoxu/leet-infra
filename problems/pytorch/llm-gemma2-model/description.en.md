# Full Gemma2 Model: Config-Driven Dimensions, Alternating Layer Types and a Tied LM Head

## Problem statement

Previous problems implemented RMSNorm, RoPE, GQA, GeGLU and the Gemma2 decoder layer one at a time; this one assembles them into a causal language model that can be compared **end to end against a complete `transformers` model**. The difference is that a real inference stack receives a config, so the model must **derive** `head_dim`, the KV head count and each layer's mask type from that config, and must **tie** the LM head to the input embedding when `tie_word_embeddings=True`.

Gemma2 also has four model-level details that only show up once everything is stacked: the input embedding is scaled by \(\sqrt{h}\), even/odd layers alternate sliding-window and full attention, attention logits get a tanh softcap **before** the mask is added, and the final logits get a second softcap. None of these belong to a single layer; getting them right is the point of this problem.

Data flow:

$$ \text{ids} \xrightarrow{\times\sqrt{h}} X_0 \xrightarrow{\text{layer}_1}\cdots\xrightarrow{\text{layer}_L} X_L \xrightarrow{\text{RMSNorm}} \hat X \xrightarrow{\text{LM head}} \text{logits} \xrightarrow{\text{softcap}} \hat{\text{logits}} $$

## Rules

- The full signature, every `config` key and every default live only in the starter docstring; that docstring is the judging contract.
- **Dimension derivation**: `head_dim` defaults to `hidden_size // num_attention_heads` (an explicit value wins), `num_key_value_heads` defaults to `num_attention_heads`, and `num_hidden_layers` defaults to `len(layers)`. The derived values must agree with the weight shapes, otherwise raise ValueError — no silent truncation and no quietly running with the wrong dimension.
- **Layer-type derivation**: `layer_types` defaults to "layer 0 is `sliding_attention`, layer 1 is `full_attention`, alternating". A sliding layer additionally requires \(\text{kv} > \text{q} - \text{sliding\_window}\) (the window includes the current token).
- **LM head tying**: when `tie_word_embeddings=True` (the default) `lm_head_weight` must be `None` and the LM head reuses `embed_weight`; when `False` a separate LM head is required.
- Per-layer order: RMSNorm (gain applied as \(1+w\)) → q/k/v projections → split heads → RoPE → GQA repeat for k/v → \(\text{logits}=qk^{\mathsf T}/\sqrt{\text{query\_pre\_attn\_scalar}}\) → tanh softcap (can be disabled in the config) → mask to \(-\infty\) → softmax → multiply by v → merge heads → o_proj → **RMSNorm, then add the residual**; then RMSNorm → GeGLU (`gelu_tanh`) → down → **RMSNorm, then add the residual**.
- A final RMSNorm feeds the LM head; when `final_logit_softcapping` is not `None`, apply \(\text{cap}\cdot\tanh(\text{logits}/\text{cap})\).
- Compute internally in float64 and return float32 logits of shape `(B, S, vocab)`.

## Example

A one-layer model with `hidden=2`, `head_dim=2`, a single head, window 1 (attend only to itself), identity projections, all four norm gains set to 0 (so \(1+w=1\)), and a tied LM head:

```
embed      = [[1, 1], [1, -1], [2, 0]]
input_ids  = [[0, 1, 2]]
config     = {num_attention_heads: 1, head_dim: 2, sliding_window: 1,
              query_pre_attn_scalar: 1, tie_word_embeddings: true}

logits[0] = [2.000000, 0.000000, 2.000000]
logits[1] = [0.534581, 1.927232, 2.461812]
logits[2] = [1.414214, 1.414214, 2.828427]
```

Token 0 can be checked by hand: the scaled embedding is \([1.414,1.414]\), RMSNorm normalizes it to \([1,1]\), attention sees only itself and returns its own value, the residual gives \([2.414,2.414]\), GeGLU adds 1, the final norm gives \([1,1]\), and multiplying by the tied head \([1,1]^{\mathsf T}\) produces \([2,0,2]\) — the third column comes from `embed[2] = [2,0]`.

## Constraints

- `embed_weight`, `final_norm_weight`, all 11 per-layer weights and the optional LM head are float32/float64 tensors, all finite, and their shapes must match the derived dimensions exactly.
- `input_ids` must be a 2-D int32/int64 tensor, non-empty, with values in \([0,\text{vocab})\).
- `config` accepts only the keys listed in the docstring; an unknown key raises ValueError. `num_attention_heads` is required.
- `head_dim` must be even; `num_attention_heads` must be divisible by `num_key_value_heads`; if `hidden_size` is not divisible by `num_attention_heads` and no `head_dim` is given, raise ValueError.
- `sliding_window >= 1` and integral; `query_pre_attn_scalar`, `rms_norm_eps` and `rope_theta` are positive finite reals; both softcaps are positive finite reals or `None`.
- Tying must agree with whether `lm_head_weight` is passed, otherwise raise ValueError.
- Time \(O\!\left(L\cdot\left(B S h (H_q+2H_{kv})d + B H_q S^2 d + 2 B S h I\right)\right)\), plus \(O(B H_q S^2)\) for the attention weights of full-attention layers.
