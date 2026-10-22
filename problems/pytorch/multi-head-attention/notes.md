# Multi-Head Attention 详解

## 定义与动机

把 $d_{model}$ 拆成 $h$ 个头，每头维度 $d_k=d_{model}/h$，各自独立做 SDPA，再拼接：

$$ \text{MHA}=\text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O $$
$$ \text{head}_i=\text{Attn}(xW_i^Q,\ xW_i^K,\ xW_i^V) $$

多头让模型在**不同表示子空间**并行关注不同位置（语法、共指、远程依赖…），单头只能学一种注意模式。总计算量与单头 $d_{model}$ 版相当，但表达更强。

## 本关实现要点

给定 `q,k,v` 已是形状 $(B, h, L, d_k)$ 的多头张量（或需 reshape/transpose 到该形状）。关键：

1. **头的维顺序**：通常是 $(B,h,L,d_k)$，注意 $QK^\top$ 在 `(-2,-1)` 上做：`q @ k.transpose(-2,-1)` → $(B,h,L_q,L_k)$。把 batch/head 轴搞错会让不同头互相串。
2. **缩放用每头维度 $d_k$**，不是 $d_{model}$：除 $\sqrt{d_k}$。
3. **mask 语义**与 SDPA 相同（torch bool True=keep，整行屏蔽输出 0，见 SDPA 关）。

## 结构变体（视野）

- **因果多头**：每头上三角 mask（见「causal mask」关）。
- **MQA/GQA**：所有/部分头共享同一份 $K,V$，省 KV-Cache 显存，是 LLaMA 等大模型推理提速关键。
- **交叉注意力**：$Q$ 来自目标、$K,V$ 来自源，实现同构。

## 对拍参考解

可用 `_ref_mha`（手导向：缩放点积 + keep-mask + softmax@V，按头堆叠），或对齐 `F.multi_head_attention_forward`（其 `attn_mask` 形状/语义需按 `(L_q, L_k)` 或 3D 提供）。
