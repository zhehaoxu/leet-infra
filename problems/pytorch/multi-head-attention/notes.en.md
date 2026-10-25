# Multi-Head Attention in Detail

## Definition and Motivation

Split $d_{model}$ into $h$ heads, each of dimension $d_k=d_{model}/h$; each one runs SDPA independently and the results are concatenated:

$$ \text{MHA}=\text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O $$
$$ \text{head}_i=\text{Attn}(xW_i^Q,\ xW_i^K,\ xW_i^V) $$

Multi-head attention lets the model attend to different positions in parallel across **different representation subspaces** (syntax, coreference, long-range dependencies...), whereas a single head can only learn one attention pattern. The total compute is comparable to the single-head $d_{model}$ version, but the expressiveness is stronger.

## Key Points for This Level

Here `q,k,v` are already multi-head tensors of shape $(B, h, L, d_k)$ (or must be reshaped/transposed into that shape). The key points:

1. **Dimension order of the heads**: usually $(B,h,L,d_k)$; note that $QK^\top$ is computed over `(-2,-1)`: `q @ k.transpose(-2,-1)` → $(B,h,L_q,L_k)$. Mixing up the batch/head axes makes different heads bleed into one another.
2. **Scale by the per-head dimension $d_k$**, not $d_{model}$: divide by $\sqrt{d_k}$.
3. **Mask semantics** are the same as in SDPA (torch bool True=keep; a fully masked row outputs 0, see the SDPA level).

## Structural Variants (For Context)

- **Causal multi-head**: an upper-triangular mask per head (see the "causal mask" level).
- **MQA/GQA**: all/some heads share the same $K,V$, saving KV-Cache memory; this is the key to faster inference in large models such as LLaMA.
- **Cross-attention**: $Q$ comes from the target while $K,V$ come from the source; the implementation is structurally identical.

## Reference Solution for Cross-Checking

You can use `_ref_mha` (hand-written: scaled dot-product + keep-mask + softmax@V, stacked per head), or align with `F.multi_head_attention_forward` (whose `attn_mask` shape/semantics must be supplied as `(L_q, L_k)` or as 3D).
