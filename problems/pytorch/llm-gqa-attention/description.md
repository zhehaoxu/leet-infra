# 大模型组件：GQA 因果注意力

## 题目描述

解码器自注意力，含现代大模型的两个关键设计：**分组查询注意力（GQA）** 与 **因果（单向）掩码**。LLaMA-2-70B、Qwen、Mistral、DeepSeek 都用 GQA：多个 query head 共享一组 key/value head，大幅压缩 KV cache。

## 规则

输入 `hidden_states`（形状 $(B,S,\text{hidden})$）、已算好的 RoPE `cos`/`sin`（形状 $(B,S,\text{head\_dim})$）、以及投影权重（线性用 $[\text{out},\text{in}]$ 布局、`F.linear(x,W)=x@W^\top$）：

- `q_weight` $(H_q\!\cdot\!d,\ \text{hidden})$、`q_bias` $(H_q\!\cdot\!d)$
- `k_weight` $(H_k\!\cdot\!d,\ \text{hidden})$、`k_bias` $(H_k\!\cdot\!d)$（Qwen2 的 q/k/v **带 bias**）
- `v_weight` $(H_k\!\cdot\!d,\ \text{hidden})$、`v_bias` $(H_k\!\cdot\!d)$
- `o_weight` $(\text{hidden},\ H_q\!\cdot\!d)$（输出投影**无 bias**）

其中 $H_q$ 为 query head 数、$H_k$ 为 kv head 数（$H_k \mid H_q$）、$d=$`head_dim`。返回 `attn_output`（形状 $(B,S,\text{hidden})$）。步骤：

1. 投影并拆头：$q=(xW_q^\top+b_q).\text{view}(B,S,H_q,d).transpose(1,2)$，$k,v$ 同理（$H_k$ 头）。
2. 施加 RoPE（与 `llm-rope-apply` 同：`rotate_half`、cos/sin 在 head 轴广播）。
3. **GQA**：把每组 $H_q/H_k$ 个 query 共享的 $k,v$ 复制（`repeat_kv`）到 $H_q$ 头。
4. 缩放点积 + 因果掩码 + softmax：
   $$ A=\text{softmax}_{\mathbb{R}}\!\Big(\tfrac{QK^\top}{\sqrt{d}} + M\Big),\quad M_{ij}=\begin{cases}0,&j\le i\\-\infty,&j>i\end{cases} $$
   （softmax 在 float32 下算再转回）
5. 合并 $O=(AV).\text{transpose}(1,2).\text{reshape}(B,S,H_q d)$，再 $+\;O W_o^\top$（无 bias）。

> 判题用 `transformers` 的 `Qwen2Attention`（**eager 实现**、同权重、同 cos/sin、同因果掩码）作 oracle。

## 示例

取 $d=4$、$H_q=H_k=1$、所有投影为单位阵且 bias 为 0、`cos=1, sin=0`（关闭旋转），
`hidden_states = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]`（$S=3$）时输出为：

```
[[1.0,      0.0,      0.0,      0.0],
 [0.377541, 0.622459, 0.0,      0.0],
 [0.274069, 0.274069, 0.451863, 0.0]]
```

第 1 行只看到自己 → 原样；后续行按因果权重混合可见的历史 token。

## 约束

- $H_k$ 整除 $H_q$；$d$ 为偶数；dtype float32
- head 数与 `head_dim` 可由权重形状与 `cos` 推出（`head_dim = cos.shape[-1]`，$H_q=q\_weight.shape[0]/d$）
- `torch.allclose` 相对容差 $10^{-5}$ 对拍 `transformers` eager；不修改输入
