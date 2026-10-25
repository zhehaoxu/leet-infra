# LLM Components: GQA Causal Attention

## Problem

Decoder self-attention, incorporating two key designs of modern LLMs: **grouped-query attention (GQA)** and a **causal (unidirectional) mask**. LLaMA-2-70B, Qwen, Mistral, and DeepSeek all use GQA: multiple query heads share one set of key/value heads, greatly compressing the KV cache.

## Rules

Inputs: `hidden_states` (shape $(B,S,\text{hidden})$), precomputed RoPE `cos`/`sin` (shape $(B,S,\text{head\_dim})$), and the projection weights (linear layers use the $[\text{out},\text{in}]$ layout, `F.linear(x,W)=x@W.T`):

- `q_weight` $(H_q\!\cdot\!d,\ \text{hidden})$, `q_bias` $(H_q\!\cdot\!d)$
- `k_weight` $(H_k\!\cdot\!d,\ \text{hidden})$, `k_bias` $(H_k\!\cdot\!d)$ (Qwen2's q/k/v **have bias**)
- `v_weight` $(H_k\!\cdot\!d,\ \text{hidden})$, `v_bias` $(H_k\!\cdot\!d)$
- `o_weight` $(\text{hidden},\ H_q\!\cdot\!d)$ (the output projection has **no bias**)

Here $H_q$ is the number of query heads, $H_k$ the number of kv heads ($H_k \mid H_q$), and $d=$`head_dim`. Return `attn_output` (shape $(B,S,\text{hidden})$). Steps:

1. Project and split heads: $q=(xW_q^\top+b_q).\text{view}(B,S,H_q,d).transpose(1,2)$, and likewise for $k,v$ ($H_k$ heads).
2. Apply RoPE (same as `llm-rope-apply`: `rotate_half`, with cos/sin broadcast over the head axis).
3. **GQA**: copy (`repeat_kv`) the $k,v$ shared by each group of $H_q/H_k$ queries out to $H_q$ heads.
4. Scaled dot product + causal mask + softmax:
   $$ A=\text{softmax}_{\mathbb{R}}\!\Big(\tfrac{QK^\top}{\sqrt{d}} + M\Big),\quad M_{ij}=\begin{cases}0,&j\le i\\-\infty,&j>i\end{cases} $$
   (softmax is computed in float32 and then cast back)
5. Merge $O=(AV).\text{transpose}(1,2).\text{reshape}(B,S,H_q d)$, then $+\;O W_o^\top$ (no bias).

> Grading uses `Qwen2Attention` from `transformers` (**eager implementation**, same weights, same cos/sin, same causal mask) as the oracle.

## Examples

With $d=4$, $H_q=H_k=1$, all projections as identity matrices with bias 0, `cos=1, sin=0` (rotation disabled),
for `hidden_states = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]` ($S=3$) the output is:

```
[[1.0,      0.0,      0.0,      0.0],
 [0.377541, 0.622459, 0.0,      0.0],
 [0.274069, 0.274069, 0.451863, 0.0]]
```

Row 1 sees only itself → unchanged; later rows blend the visible history with causal weights.

## Constraints

- $H_k$ divides $H_q$; $d$ is even; dtype float32
- The head counts and `head_dim` can be derived from the weight shapes and `cos` (`head_dim = cos.shape[-1]`, $H_q=q\_weight.shape[0]/d$)
- `torch.allclose` with relative tolerance $10^{-5}$ against `transformers` eager; do not modify the inputs
