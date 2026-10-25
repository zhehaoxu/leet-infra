# LLM Components: DeepSeek Multi-head Latent Attention (MLA)

## Problem

**MLA (Multi-head Latent Attention, DeepSeek-V2/V3)** replaces the conventional
KV cache with **low-rank latent compression**: instead of caching a full K/V per head, it compresses Key/Value into a short latent vector
$c_{KV}$ shared by all heads, which is expanded again with an up-projection matrix during the attention computation. Positional encoding travels on a separate **decoupled RoPE**
branch $k_{pe}$—it is the only component carrying position, and all heads share the same copy. The KV cache therefore stores only
$c_{KV}$ and $k_{pe}$, greatly reducing GPU memory for long contexts; this is the advanced step of the attention family after `llm-gqa-attention` in this collection.

Data flow (matching the `q_lora_rank` path, non-interleaved RoPE, of HF's `DeepseekV3Attention.forward`):

$$ q_{\text{latent}} = \text{RMSNorm}(W_{qa}\,x),\qquad q = W_{qb}\,q_{\text{latent}} \rightarrow (\text{heads},\ d_{qk}) $$
$$ [\,c_{KV}^{\text{raw}},\ k_{pe}\,] = W_{kva}\,x \rightarrow(\text{heads}=1,\ d_{kv}+d_{rope}),\qquad c_{KV}=\text{RMSNorm}(c_{KV}^{\text{raw}}) $$
$$ (\tilde q_{rot},\ \tilde k_{pe}) = \text{RoPE}(q_{rot},\ k_{pe}) \quad\text{— applied only to the rope components} $$
$$ [\,k_{nope},\ v\,] = W_{kb}\,c_{KV}\rightarrow(\text{heads},\ d_{nope}+d_v) $$
$$ q=[q_{nope}\,\|\,\tilde q_{rot}],\quad k=[k_{nope}\,\|\,\tilde k_{pe}],\quad o = \text{Attn}(q,k,v) \rightarrow W_o $$

where $d_{qk}=d_{nope}+d_{rope}$; attention scaling uses $1/\sqrt{d_{qk}}$, with a causal $-\infty$ mask and fp32 softmax.

## Rules

Inputs: `hidden_states` $(B,S,\text{hidden})$, precomputed `cos`/`sin` $(B,S,d_{rope})$, and all weights
(linear layers use the $[\text{out},\text{in}]$ layout, `F.linear(x,W)=x@Wᵀ`):

- `q_a_proj_weight` $(d_q,\text{hidden})$, `q_a_norm_weight` $(d_q,)$, `q_b_proj_weight` $(n_h\cdot d_{qk},\ d_q)$
- `kv_a_proj_weight` $(d_{kv}+d_{rope},\ \text{hidden})$, `kv_a_norm_weight` $(d_{kv},)$, `kv_b_proj_weight` $(n_h\cdot(d_{nope}+d_v),\ d_{kv})$
- `o_proj_weight` $(\text{hidden},\ n_h\cdot d_v)$
- integers `n_heads, nope_head_dim, rope_head_dim, v_head_dim`; `d_q=` `q_a_proj_weight.shape[0]`, $d_{kv}=$ `kv_a_norm_weight.shape[0]`, `rms_eps` defaults to `1e-6`

Strictly in order:

1. `q_latent = RMSNorm(F.linear(hs, q_a_proj_weight), q_a_norm_weight)`; `q = F.linear(q_latent, q_b_proj_weight)` → view `(B,S,n_h,d_qk)` → `transpose(1,2)` → split along the last dim into `q_nope(d_nope)` and `q_rot(d_rope)`
2. `compressed = F.linear(hs, kv_a_proj_weight)` → split into `c_kv_raw(d_kv)` and `k_pe(d_rope)`; `c_kv = RMSNorm(c_kv_raw, kv_a_norm_weight)`, treated as single-head `(B,1,S,d_kv)` and `(B,1,S,d_rope)`
3. **Apply RoPE only to `q_rot` and `k_pe`**: `x*cos.unsqueeze(1) + rotate_half(x)*sin.unsqueeze(1)` (`rotate_half` as in `llm-rope-apply`)
4. `query = cat([q_nope, q_rot], -1)` $(B,n_h,S,d_{qk})$
5. `kv = F.linear(c_kv, kv_b_proj_weight)` → view `(B,S,n_h,d_nope+d_v)` → `transpose(1,2)` → split into `k_nope(d_nope)` and `v(d_v)`; broadcast `k_pe` to $n_h$ heads, then `key = cat([k_nope, k_pe], -1)`
6. `score = query @ keyᵀ * d_qk**-0.5 + causal -inf`, `softmax(dim=-1, dtype=fp32)`, `out = score @ v` → `transpose`→reshape `(B,S,n_h·d_v)` → `F.linear(out, o_proj_weight)`, return $(B,S,\text{hidden})$

> `num_key_value_heads == num_attention_heads`, i.e. all heads share the same compressed latent and **no repeat_kv is needed**.
> Grading uses `DeepseekV3Attention` from `transformers` (**eager**, `rope_interleave=False`, same weights, same cos/sin, same causal mask) as the oracle.

## Examples

With `B=1,S=3,hidden=4`, `n_heads=2, nope=2, rope=2, v=2`, `q_lora=3, kv_lora=4`,
after `torch.manual_seed(11)`, generating `hidden_states` and each projection weight with `randn` in the shapes listed above,
both norm gains all `1`, and `cos=1, sin=0`, the output $(B,S,4)$ is:

```
[[[ 0.877777,  1.533879,  1.298111,  0.869258],
  [-1.622245, -1.653416, -3.259178, -2.022986],
  [-2.708383, -1.197325, -0.054776, -0.642115]]]
```

## Constraints

- `rope_head_dim` is even (`rotate_half` splits it in half); $d_{qk}=d_{nope}+d_{rope}$, and scaling uses $d_{qk}$ rather than $d_{nope}$
- RMSNorm computes `rsqrt(mean(x²)+eps)` in fp32, casts back to the original dtype, then multiplies by the gain (as in `llm-rmsnorm`)
- RoPE acts only on the rope components; the `nope` content components carry **no position**; `k_pe` is identical for all heads
- `torch.allclose` with relative/absolute tolerance $10^{-5}$ against `transformers` eager; do not modify the inputs
