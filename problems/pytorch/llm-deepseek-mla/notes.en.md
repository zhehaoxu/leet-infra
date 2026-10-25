# DeepSeek MLA Multi-head Latent Attention in Depth

## Why latent compression (the motivation for MLA)

Standard MHA/GQA must cache $H\times d_k$ keys and values per layer per token, and with long contexts the KV cache eats the bulk of GPU memory.
DeepSeek-V2/V3's **MLA** first **down-projects** Key/Value into a low-dimensional latent
$c_{KV}\in\mathbb{R}^{d_{kv}}$ shared by all heads ($d_{kv}\ll H\cdot d_k$), caches **only $c_{KV}$**, and when needed
**up-projects** it through `kv_b_proj` into per-head $k_{nope},v$. Memory drops from $O(H d_k)$ to $O(d_{kv})$.
In this toy, `kv_a_proj` compresses `hidden` to `kv_lora+rope` and the attention expands it internally—that expansion is exactly what you write.

The query side can likewise be low-rank (`q_a_proj→RMSNorm→q_b_proj`); this problem takes the `q_lora_rank` path.

## Decoupled RoPE

Latent compression would also compress away positional information, making attention lose position. MLA's answer: route the components that RoPE needs through a **separate**
uncompressed branch $k_{pe}$ (the last `rope_head_dim` dims of the `kv_a_proj` output), which is concatenated with the compressed $c_{KV}$ to form the full key:
$k = [\,k_{nope}\,\|\,\tilde k_{pe}\,]$. Key points:

- **RoPE acts only on the rope components** (`q_rot`, `k_pe`); the `nope` content components add no position—`q = [q_nope‖q̃_rot]`.
- `k_pe` is **shared across heads**: after `expand` to $n_h$ heads, the rope segment is identical in every head, and only `k_nope` differs per head.
- `rope_head_dim` must be even (`rotate_half` splits it in half), and here HF rotary's `head_dim` equals `rope_head_dim`.

## Relation to GQA

In `llm-gqa-attention`, K/V have multiple independent heads and `repeat_kv` broadcasts the $n_{kv}$ groups to $n_q$ heads.
MLA is the reverse: there is **only one shared latent**, expanded by `kv_b_proj` into $n_H$ heads of $k_{nope}/v$, so
`num_key_value_heads == num_attention_heads`, `num_key_value_groups=1`, and **repeat_kv is unnecessary**.
Scaling still uses the inverse square root of the full $d_{qk}=d_{nope}+d_{rope}$—do not divide by $d_{nope}$ alone.

## RMSNorm details

Both norms (`q_a_norm`, `kv_a_norm`) have the same semantics as `llm-rmsnorm`: `x*rsqrt(mean(x²)+eps)` in fp32,
cast back to the original dtype, then multiply element-wise by the gain, **without subtracting the mean**. `kv_a_norm` acts on the $d_{kv}$ dimension of $c_{KV}$.

## Common mistakes to catch

- Forgetting to RMSNorm $c_{KV}$ or $q_{latent}$ → large deviation from the oracle.
- Applying RoPE to the `nope` components by mistake, or omitting it on the rope components → every position-dependent case fails.
- Concatenating `k_pe` without `expand`ing it to $n_h$ heads → shape mismatch.
- Using $d_{nope}$ or $d_v$ for scaling instead of $d_{qk}$.

## Related

- Upstream: `llm-rmsnorm`, `llm-rope-apply`, `llm-gqa-attention`.
- Cross-check: `transformers` `DeepseekV3Attention` (`attn_implementation="eager"`, `rope_interleave=False`,
  `num_key_value_heads=num_attention_heads`, `head_dim=rope_head_dim`), same weights, same cos/sin, same causal mask,
  `allclose` at $10^{-5}$. `rope_interleave=True` (the HF default) is an equivalent alternative dimension layout; this problem fixes the non-interleaved one.
- arXiv: DeepSeek-V2 2405.04434, DeepSeek-V3 2412.19437 (MLA / decoupled RoPE).
