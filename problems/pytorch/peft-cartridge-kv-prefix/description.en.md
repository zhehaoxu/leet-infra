# PEFT: Cartridge's Directly Parameterized KV-Prefix Adapter

## Problem Description

**Cartridge** (arXiv:2506.06266, Stanford Hazy Research, now in HF PEFT `tuners/cartridge`, PR #2953) stores a **learnable compressed KV cache** directly as parameters: unlike prefix tuning, which learns an MLP that generates the prefix, Cartridge's prefix KV is a trainable embedding table injected straight into attention with prefix semantics; the paper recommends freezing the first token as an **attention sink** for training stability.

The deterministic ledger of the official PEFT implementation (all anchors re-computable offline):

```
# 1) pack a legacy KV cache into prompt embeddings (utils.py)
packed[:, 2*l]     = k_l[0, :, :NVT, :].transpose(0, 1)   # (NVT, H, D)
packed[:, 2*l + 1] = v_l[0, :, :NVT, :].transpose(0, 1)
pe = packed.reshape(NVT, L*2*H*D)                          # interleaved K/V per layer

# 2) CartridgeEncoder (model.py): frozen segment + trainable segment
frozen_embedding   ~ N(0, 1)   # requires_grad=False, drawn FIRST from the RNG stream
trainable_embedding ~ N(0, 1)  # requires_grad=True, drawn SECOND
weight = cat([frozen, trainable], dim=0)                   # (NVT, L*2*token_dim)

# 3) load_prompt_embeddings: split rows by num_frozen_tokens
# 4) forward: ignore token ids, weight.unsqueeze(0).expand(batch, -1, -1)
# 5) compose: cat(prompt_embeddings, dim=0), frozen count from the FIRST adapter
```

Three key conventions:

1. **Interleaved layout**: in the flattened row, column index \(2l\cdot HD + h\cdot D + d\) holds the **K** component of layer \(l\), head \(h\), virtual token \(t\); \(2l+1\) holds the **V** component — layers interleaved K/V, heads before head_dim (C order).
2. **Frozen first**: the frozen segment is the **leading** `num_frozen_tokens` rows (the attention sink) with `requires_grad=False`; `reset_parameters` initializes **frozen before trainable** with `normal_` (std=1) in the official order — the order on the same RNG stream is not interchangeable.
3. **Compose takes the first adapter's frozen count**: when concatenating adapters, `num_frozen_tokens` comes only from the first adapter (a single attention-sink prefix); compatibility requires `peft_type="CARTRIDGE"` plus agreement on task_type / token_dim / num_layers / num_attention_heads / num_transformer_submodules.

## Task

Implement three groups of entry points:

- `prompt_embeddings_from_past_key_values(past_key_values, *, num_virtual_tokens)`: the input is a per-layer list of `(k, v)` tuples (each `[1, H, S, D]`). Validation (all raise `ValueError`): empty cache, rank != 4, batch != 1, `seq_len < NVT`, per-layer head/head_dim inconsistency, per-layer cache length shortfall. Returns `(NVT, L*2*H*D)`.
- `CartridgeEncoder(num_virtual_tokens, num_layers, token_dim, num_frozen_tokens=1, dtype=torch.float32)`: validates `0 <= num_frozen_tokens <= num_virtual_tokens` (out of range raises `ValueError`); the `weight` property (frozen-first concatenation; when there is no frozen segment it IS the trainable parameter itself); `load_prompt_embeddings` (wrong rows/dims raise `ValueError`, splits by the frozen count); `forward(prefix_tokens)` (ignores ids, expands the batch).
- `compose_cartridge_adapters(adapters)`: the input is a list of `{"prompt_embeddings": tensor, "config": dict}`. Fewer than 2, non-`"CARTRIDGE"` `peft_type`, or incompatible compat attributes raise `ValueError`; returns `(concatenated embeddings, first adapter's num_frozen_tokens)`.

## Example

\(L=3, H=4, D=5, S=10, NVT=6\):

```
prompt_embeddings_from_past_key_values(cache, num_virtual_tokens=6).shape -> (6, 120)
pe[3, 2*0*4*5 + 1*5 + 4] == cache[0][0][0, 1, 3, 4]        # layer-0 K, head 1, token 3, dim 4
enc = CartridgeEncoder(6, 3, 20, num_frozen_tokens=1)       # hidden = 3*2*20 = 120
enc.weight.shape -> (6, 120); enc.forward(empty ids (8,6)).shape -> (8, 6, 120)
compose([pa(4,120), pb(4,120)]) -> ((8,120), 1)
```

## Constraints

- Packing is a pure layout permutation: no value changes, only slicing/transposing/concatenating (`contiguous` or not does not affect the result).
- Packing takes the **first** `num_virtual_tokens` cached tokens (truncation from the front, not the tail).
- The encoder's `dtype` argument only affects storage precision; RNG anchoring uses the official float32 path (`torch.empty(...).normal_()`).
- Compose concatenates in the adapters' list order; the frozen count comes from the first adapter even if later configs differ.
