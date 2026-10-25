# LLM Components: Decoder Layer (Pre-Norm, Dual Residual)

## Problem

Assemble the previous three building blocks into **one layer** of a Transformer decoder. Modern LLMs (Qwen, LLaMA, Mistral, DeepSeek) all use **Pre-Norm**: the norm is placed **before** the sublayer, and the residual adds the sublayer output back to the input. The data flow of one layer:

$$ h = x + \text{Attn}(\text{RMSNorm}(x)) $$
$$ y = h + \text{MLP}(\text{RMSNorm}(h)) $$

## Rules

Inputs: `hidden_states` $(B,S,\text{hidden})$, precomputed RoPE `cos`/`sin` $(B,S,\text{head\_dim})$, and all weights of two norm groups, attention, and the MLP (linear weights use the $[\text{out},\text{in}]$ layout, `F.linear(x,W)=x@Wᵀ`):

- `input_ln_weight`, `post_attention_ln_weight`: the gains of the two `RMSNorm`s ($(\text{hidden})$)
- Attention: `q_weight/q_bias/k_weight/k_bias/v_weight/v_bias/o_weight` (same semantics as `llm-gqa-attention`; q/k/v have bias, o has no bias)
- MLP: `gate_weight/up_weight/down_weight` (same semantics as `llm-swiglu-mlp`; all three are bias-free)

`head_dim = cos.shape[-1]`, and the numbers of query/kv heads are inferred from the weight row counts. `rms_eps` defaults to `1e-6`. Return the layer output (shape $(B,S,\text{hidden})$). Strictly in order:

1. `residual = x`; `x1 = RMSNorm(x, input_ln_weight)`
2. `a = GQA causal attention(x1, cos, sin, ...)` (RoPE→repeat_kv→$QK^\top/\sqrt{d}$→causal $-\infty$ mask→fp32 softmax→$AV$→o_proj)
3. `h = residual + a`
4. `residual = h`; `h1 = RMSNorm(h, post_attention_ln_weight)`
5. `m = SwiGLU-MLP(h1)`: `down(silu(gate)·up)`
6. Return `residual + m`

> Grading uses `Qwen2DecoderLayer` from `transformers` (**eager attention**, same weights, same cos/sin, same causal mask) as the oracle.

## Examples

With $d=4$, $H_q=H_k=1$, `hidden=inter=4`, all projections as identity matrices, biases 0, both norm gains all 1, and `cos=1, sin=0`,
for `hidden_states = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]` ($S=3$) the output is:

```
[[6.523182, 0.0,      0.0,      0.0],
 [0.254472, 6.255613, 0.0,      0.0],
 [0.227636, 0.227636, 6.043899, 0.0]]
```

The residual adds the original tokens on top of the attention/MLP results, so the diagonal entries are far larger than 1.

## Constraints

- $H_k$ divides $H_q$; $d$ is even; dtype float32
- The residual and the norm placement (**pre-norm**) cannot be swapped; the residual is an addition, not a concatenation
- `torch.allclose` with relative tolerance $10^{-5}$ against `transformers` eager; do not modify the inputs
