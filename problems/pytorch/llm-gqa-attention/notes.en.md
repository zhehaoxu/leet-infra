# GQA Causal Attention in Depth

## From MHA to MQA/GQA

In standard multi-head attention (MHA), every query head has its own set of key/value heads, so the KV cache grows with layers × heads and neither memory nor bandwidth can keep up during long-context inference.

- **MQA** (Shazeer 2019): all query heads **share one set** of kv heads ($H_k=1$). Maximum memory savings, but a slight quality drop.
- **GQA** (Ainslie 2023): the compromise—split the $H_q$ query heads into $H_k$ groups, with **one shared** kv set **within each group**. LLaMA-2-70B, Qwen, Mistral, and DeepSeek all use it.

This problem expresses the grouping naturally through weight shapes: `q_weight` has $H_q d$ rows, `k/v_weight` have $H_k d$ rows, and $H_k \mid H_q$. When $H_k=1$ it degenerates to MQA (the tests include a dedicated case).

## The correspondence in repeat_kv

HF's `repeat_kv` expands $k,v$ from $H_k$ heads to $H_q$ heads via `x[:,:,None].expand(B,H_k,n_rep,S,d).reshape(B,H_k*n_rep,S,d)`—**each group of $n_\text{rep}=H_q/H_k$ adjacent query heads shares the same kv head**. It must be written with `repeat_interleave`; using `repeat` (tiling everything at once) mismatches heads. The tests cover this with `nh=4,nkv=2` and `nkv=1` against the oracle.

## Causal mask: -inf vs finfo.min

Causality = position $i$ may only see $j\le i$. The approach is to set the upper triangle to $-\infty$ and then apply softmax, so future tokens get zero weight.

- This solution uses `torch.full(...,float("-inf")).triu(1)`; HF uses `finfo.min` (a very large finite negative number).
- Inside softmax the two agree ($\exp(-\infty)=0$; $\exp(\text{finfo.min})\approx 0$); the difference is only in the last floating-point bits, well within the $10^{-5}$ tolerance. The scratch cross-check gives a maxdiff of about $3\times10^{-7}$, which is acceptable.
- The first token attends only to itself → its output equals its $v$ (with identity projections, the input itself); `check_first_row_only_self` verifies exactly this.

## Other semantic pitfalls

- **q/k/v have bias, o_proj does not**: this is the case for Qwen2 (same as Llama). Omitting the bias will not match the oracle.
- **Scaling is $\sqrt{d}$** (head_dim), not hidden. `scaling = head_dim**-0.5`.
- **Softmax is computed in float32 and then cast back**: consistent with HF's `eager_attention_forward`, avoiding underflow in half precision.
- **RoPE acts only on q/k, never on v**, and is broadcast over the head axis with `cos.unsqueeze(1)`—exactly the same rule as `llm-rope-apply`.

## Related

- Upstream: `llm-rope-apply` (the rotation here is that problem; this solve carries its own `_rotate_half` so it can be evaluated independently).
- Downstream: `llm-decoder-layer` (chains this attention + RMSNorm + SwiGLU MLP into one pre-norm dual-residual layer).
- Cross-check: `transformers` `Qwen2Attention` (**`attn_implementation="eager"` is mandatory**; the default sdpa uses a fused kernel whose numerical path differs slightly from a hand-written eager version), same weights, same cos/sin, same causal mask, `allclose` at $10^{-5}$.
