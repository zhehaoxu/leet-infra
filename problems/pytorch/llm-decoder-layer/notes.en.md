# Decoder Layer in Depth

## Pre-Norm vs Post-Norm

The original Transformer is **Post-Norm**: `Norm(x + Sublayer(x))`. Modern LLMs almost all use **Pre-Norm**: `x + Sublayer(Norm(x))`—the norm comes **before** the sublayer and the residual adds the original $x$ directly. Pre-Norm turns the residual stream into a "clean" identity highway, so gradients flow freely and deep stacks still train stably.

One layer is just two Pre-Norm sublayers in series:

```
h = x + Attn(RMSNorm(x, input_ln_weight))
y = h + MLP (RMSNorm(h, post_attention_ln_weight))
```

The key point: **the residual adds the tensor that entered the sublayer, and the norm only acts on the sublayer's input**. Writing it the other way round as `RMSNorm(x + Attn(x))` is wrong.

## Assembling the finished blocks

This layer introduces no new math; it simply wires the previous three problems together in the order of HF's `Qwen2DecoderLayer.forward`:

- `RMSNorm` (`llm-rmsnorm`) — two places, each with its own gain.
- `GQA causal attention` (`llm-gqa-attention`) — including RoPE, repeat_kv, the $-\infty$ mask, fp32 softmax, and o_proj without bias.
- `SwiGLU MLP` (`llm-swiglu-mlp`) — `down(silu(gate)·up)`, all three projections bias-free.

`cos/sin` are computed upstream (by the model-level `RotaryEmbedding`) and passed in; the layer only applies them. `head_dim` is taken from `cos.shape[-1]`, and head counts are inferred from the projection weight row counts.

## Property-based cases that catch mistakes

- **Residual identity**: setting `o_weight=0` (attention output becomes zero) and `gate_weight=0` (MLP becomes zero) gives $y=x$—both sublayers are bypassed by the residual. `check_residual_stream_identity` verifies exactly this and catches "forgot the residual" or "wrong norm placement".
- Cross-checked against `Qwen2DecoderLayer(eager)` with the same weights, in two configurations: GQA ($n_{kv}=2$) and MQA ($n_{kv}=1$).

## Related

- Upstream: `llm-rmsnorm`, `llm-rope-apply`, `llm-gqa-attention`, `llm-swiglu-mlp`.
- Downstream: `llm-qwen2-model`—chains embedding → N copies of this layer → final norm → lm_head into a complete model.
- Cross-check: `transformers` `Qwen2DecoderLayer` (**`attn_implementation="eager"`**), same weights, same cos/sin, same causal mask, `allclose` at $10^{-5}$.
