# 大模型组件：Decoder Layer（Pre-Norm 双残差）

## 题目描述

把前三块积木拼成 Transformer 解码器的**一层**。现代 LLM（Qwen、LLaMA、Mistral、DeepSeek）都用 **Pre-Norm**：norm 放在子层**之前**，残差把子层输出加回输入。一层的数据流：

$$ h = x + \text{Attn}(\text{RMSNorm}(x)) $$
$$ y = h + \text{MLP}(\text{RMSNorm}(h)) $$

## 规则

输入 `hidden_states` $(B,S,\text{hidden})$、已算好的 RoPE `cos`/`sin` $(B,S,\text{head\_dim})$，以及两组 norm、注意力、MLP 的全部权重（线性用 $[\text{out},\text{in}]$ 布局，`F.linear(x,W)=x@Wᵀ`）：

- `input_ln_weight`、`post_attention_ln_weight`：两个 `RMSNorm` 的 gain（$(\text{hidden})$）
- 注意力：`q_weight/q_bias/k_weight/k_bias/v_weight/v_bias/o_weight`（语义同 `llm-gqa-attention`，q/k/v 带 bias、o 无 bias）
- MLP：`gate_weight/up_weight/down_weight`（语义同 `llm-swiglu-mlp`，三线无 bias）

`head_dim = cos.shape[-1]`，query/kv 头数由权重行数推出。`rms_eps` 默认 `1e-6`。返回该层输出（形状 $(B,S,\text{hidden})$）。严格按序：

1. `residual = x`；`x1 = RMSNorm(x, input_ln_weight)`
2. `a = GQA因果注意力(x1, cos, sin, ...)`（RoPE→repeat_kv→$QK^\top/\sqrt{d}$→因果 $-\infty$ 掩码→fp32 softmax→$AV$→o_proj）
3. `h = residual + a`
4. `residual = h`；`h1 = RMSNorm(h, post_attention_ln_weight)`
5. `m = SwiGLU-MLP(h1)`：`down(silu(gate)·up)`
6. 返回 `residual + m`

> 判题用 `transformers` 的 `Qwen2DecoderLayer`（**eager 注意力**、同权重、同 cos/sin、同因果掩码）作 oracle。

## 示例

取 $d=4$、$H_q=H_k=1$、`hidden=inter=4`、所有投影为单位阵、bias 为 0、两个 norm gain 全 1、`cos=1, sin=0`，
`hidden_states = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]`（$S=3$）时输出为：

```
[[6.523182, 0.0,      0.0,      0.0],
 [0.254472, 6.255613, 0.0,      0.0],
 [0.227636, 0.227636, 6.043899, 0.0]]
```

残差把原始 token 与注意力/MLP 结果叠加，所以对角线远大于 1。

## 约束

- $H_k$ 整除 $H_q$；$d$ 为偶数；dtype float32
- 残差、norm 位置（**pre-norm**）不可调换；残差是加法不是拼接
- `torch.allclose` 相对容差 $10^{-5}$ 对拍 `transformers` eager；不修改输入
