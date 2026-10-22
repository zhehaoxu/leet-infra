# Scaled Dot-Product Attention 详解

## 公式

$$ \text{Attn}(Q,K,V)=\text{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}+\text{mask}\Big)V $$

它是所有注意力机制的最小内核，Transformer、cross-attention、FlashAttention 都在其上加结构。

## 为什么除以 $\sqrt{d_k}$（本关灵魂）

设 $q,k$ 各分量独立、均值 0 方差 1，则点积 $q\cdot k=\sum_{i=1}^{d_k}q_ik_i$ 方差为 $d_k$。不缩放的话，$d_k$ 越大 logits 方差越大，softmax 落入**饱和区**（近似 one-hot），梯度趋 0。除 $\sqrt{d_k}$ 把 logits 方差拉回 1，保持 softmax 工作在梯度健康的区间。

## mask 语义：torch 原生 True = 保留（本关大坑）

- PyTorch 原生 `bool mask`：**True 表示允许注意（keep）**，False 表示屏蔽。手写常被直觉误导成「True=遮掉」，方向正好相反。
- 落地做法：`scores.masked_fill(~mask, -inf)` 再加 softmax。
- **整行被屏蔽**时全 $-\infty$ → softmax `NaN`；torch 的约定是**该行输出 0**，故要 `nan_to_num(..., nan=0.0)`。

## 数值稳定

softmax 一定先减行最大值再 `exp`（见「Softmax」关）。`scores - scores.max(-1,keepdim)` 后 `exp`、归一，避免大 logits 上溢。

## 维度

$Q:(...,L_q,d_k)$、$K:(...,L_k,d_k)$、$V:(...,L_k,d_v)$；$QK^\top:(...,L_q,L_k)$，乘 $V$ 得 $(...,L_q,d_v)$。$d_k$ 只出现在 $Q,K$，$d_v$ 在 $V$——缩放只用 $d_k$。

## 对拍参考解

与 `F.scaled_dot_product_attention`（传 `attn_mask`）一致，注意后者 mask 也是加性/布尔 keep 语义。
