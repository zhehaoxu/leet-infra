# RoPE 详解

## 核心思想：用旋转编码相对位置

普通绝对位置编码是「加」一个向量；RoPE 改成对 query/key 每对相邻分量做**二维旋转**，旋转角随位置 $m$ 线性增长。这样两个向量做内积时，绝对角抵消、只剩**相对位移** $m-n$ 的贡献：

$$ \langle R_m\,q,\; R_n\,k\rangle = \langle q,\; R_{n-m} k\rangle $$

这正是大模型偏爱的：注意力天然看到「相隔多远」，且对更长序列有外推潜力。

## 实现要点

- **半维频率**：只有前 $d/2$ 个独立频率 $\text{inv\_freq}_i=base^{-2i/d}$；把角度 `cat((θ, θ))` 复制成 $d$ 维，是因为 `rotate_half` 把 $d$ 维看成 $(x_1,x_2)$ 两半配对旋转。
- **rotate_half 而非交错**：HF/LLaMA 用「前后对半」的配对方式（`[-x2, x1]`），不是相邻两两交错。配对方式错了，点积相对位置性质就破了——这也是不同实现最容易对不上的地方。
- **广播**：cos/sin 形状 $(B,S,d)$，需在 head 轴 `unsqueeze(1)` 成 $(B,1,S,d)$ 才能同时喂给 $H_q$ 个 query head 和 $H_k$ 个 key head（GQA 下二者数量不同）。
- **float32 算三角**：角度大时低精度 cos/sin 误差明显，HF 在 fp32 下算再降回。

## 基频 base 与长上下文

$base$ 越大，最低频分量波长越长，能表征的相对距离越远。Qwen/LLaMA 常把 base 从 1 万扩到 10 万甚至百万以支持长上下文（配合 YaRN 等缩放，见 rope 缩放类题目）。

## 关联

- 上游：`llm-rmsnorm`。
- 下游：`llm-gqa-attention` 里 q/k 投影后正是调用这里的旋转；`rope-embedding`（operators 合集）是更早的简化版。
- 对拍：`transformers` 的 `Qwen2RotaryEmbedding` + `apply_rotary_pos_emb`（同 position_ids/d/base），`allclose` $10^{-5}$。
