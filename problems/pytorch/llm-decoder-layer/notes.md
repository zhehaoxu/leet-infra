# Decoder Layer 详解

## Pre-Norm vs Post-Norm

原始 Transformer 是 **Post-Norm**：`Norm(x + Sublayer(x))`。现代 LLM 几乎都用 **Pre-Norm**：`x + Sublayer(Norm(x))`——norm 在子层**之前**，残差直接加原始 $x$。Pre-Norm 让残差流是一条「干净」的恒等高速路，梯度畅通、深层堆叠仍稳定训练。

一层就是两个 Pre-Norm 子层串联：

```
h = x + Attn(RMSNorm(x, input_ln_weight))
y = h + MLP (RMSNorm(h, post_attention_ln_weight))
```

关键：**残差加的是进子层前的张量、norm 只作用于进子层的输入**。写反成 `RMSNorm(x + Attn(x))` 就不对了。

## 拼装已完成的积木

这层不做新数学，只是把前三题按 HF `Qwen2DecoderLayer.forward` 的顺序接起来：

- `RMSNorm`（`llm-rmsnorm`）——两处，各用独立 gain。
- `GQA 因果注意力`（`llm-gqa-attention`）——含 RoPE、repeat_kv、$-\infty$ 掩码、fp32 softmax、o_proj 无 bias。
- `SwiGLU MLP`（`llm-swiglu-mlp`）——`down(silu(gate)·up)`，三线无 bias。

`cos/sin` 由上层（模型级 `RotaryEmbedding`）算好传入，层内只施加。`head_dim` 从 `cos.shape[-1]` 取，头数从投影权重行数推。

## 抓错的性质用例

- **残差恒等**：令 `o_weight=0`（注意力输出归零）且 `gate_weight=0`（MLP 归零），则 $y=x$——两个子层都被残差旁路。`check_residual_stream_identity` 就验这条，能抓到「忘了加残差」或「norm 位置错」。
- 与 `Qwen2DecoderLayer(eager)` 同权重对拍，GQA($n_{kv}=2$) 与 MQA($n_{kv}=1$) 两配置。

## 关联

- 上游：`llm-rmsnorm`、`llm-rope-apply`、`llm-gqa-attention`、`llm-swiglu-mlp`。
- 下游：`llm-qwen2-model`——把 embedding → N 个本层 → final norm → lm_head 串成完整模型。
- 对拍：`transformers` `Qwen2DecoderLayer`（**`attn_implementation="eager"`**），同权重同 cos/sin 同因果掩码，`allclose` $10^{-5}$。
