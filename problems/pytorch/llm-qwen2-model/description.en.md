# Assembling an LLM: the Full Qwen2 Causal Language Model

## Problem Description

Put all the previous components together into **one complete LLM**, going end-to-end from token ids to logits. This is the combined application of the five problems `llm-rmsnorm`, `llm-rope-apply`, `llm-gqa-attention`, `llm-swiglu-mlp`, and `llm-decoder-layer` — the real Qwen2-7B merely scales up the dimensions and layer counts here.

Data flow:

$$ \text{ids} \xrightarrow{\text{embed}} X_0 \xrightarrow{\text{layer}_1}\cdots\xrightarrow{\text{layer}_L} X_L \xrightarrow{\text{final RMSNorm}} \hat X \xrightarrow{\text{lm\_head}} \text{logits} $$

## Rules

Inputs:

- `input_ids` $(B,S)$: integer token ids
- `embed_weight` $(\text{vocab},\text{hidden})$: the token embedding matrix (`F.embedding` is just a row lookup)
- `layers`: a **list**, each element a dict of that layer's weights, with exactly the same keys as `llm-decoder-layer`: `input_ln_weight, post_attention_ln_weight, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, o_weight, gate_weight, up_weight, down_weight`
- `final_norm_weight` $(\text{hidden})$: the gain of the top-most RMSNorm
- `lm_head_weight` $(\text{vocab},\text{hidden})$: the output projection (**no bias**)
- `head_dim`, `base=10000.0`, `rms_eps=1e-6`

Return `logits` $(B,S,\text{vocab})$. Steps:

1. Embedding lookup: $X = \text{embed\_weight}[\text{input\_ids}]$.
2. **Generate RoPE on the fly**: positions $0..S{-}1$, $\text{inv\_freq}_i=\text{base}^{-2i/d}$, `emb=cat(freqs,freqs)`, `cos=emb.cos()`, `sin=emb.sin()`, shape $(B,S,d)$.
3. Call the **Decoder Layer** layer by layer (pre-norm → GQA causal attention → residual → pre-norm → SwiGLU MLP → residual); all layers share the same `cos/sin` and the same $-\infty$ causal mask.
4. Top-level `RMSNorm(X, final_norm_weight)`.
5. `logits = X @ lm_head_weightᵀ` (no bias).

> The judge uses `transformers`' `Qwen2ForCausalLM` (**eager attention**, same weights, same token ids) as the oracle, comparing the logits directly.

## Example

A tiny model: `vocab=4, head_dim=4, hidden=4, intermediate=4, num_layers=2, n_q=n_kv=1`, with weights initialized from a fixed seed; for `input_ids=[[0,1,2]]` the logits ($3\times4$, rounded to 6 decimal places) are approximately:

```
[[-0.070977,  0.028838, -0.070091, -0.019763],
 [-0.008341, -0.063482, -0.013246,  0.013710],
 [ 0.070258,  0.012415,  0.064580, -0.009723]]
```

(These values come from a joint cross-check of the reference implementation and the oracle; the statement shows them only to illustrate the shape/magnitude scale — the judge compares against `Qwen2ForCausalLM` via `allclose`.)

## Constraints

- $H_k$ divides $H_q$; `head_dim` is even; dtype float32
- Number of layers = `len(layers)`; they must run serially in order, each layer's output feeding the next
- In this collection `lm_head` and `embed` do **not share** weights (`tie_word_embeddings=False`)
- `torch.allclose` with relative tolerance $10^{-5}$ against `transformers` eager
