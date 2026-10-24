# 大模型组件：RoPE 旋转位置编码

## 题目描述

**RoPE（Rotary Position Embedding）** 是 LLaMA / Qwen / Mistral / DeepSeek 通用的位置编码。它不给输入「加」位置向量，而是把 query/key 的每对相邻维度按位置 $m$ **旋转一个角度**，使得注意力点积只依赖**相对位置** $m-n$。

## 规则

输入 query 张量 $q$（形状 $(B, H_q, S, d)$）、key 张量 $k$（形状 $(B, H_k, S, d)$）、位置编号 `position_ids`（形状 $(B, S)$，整数）、head 维度 $d$ 与基频 $base$；返回**旋转后**的 $(q_{rot}, k_{rot})$，各自形状不变。步骤：

1. 半维逆频率（$i = 0 \dots d/2-1$）：
   $$ \text{inv\_freq}_i = base^{-\,2i/d} $$
2. 角度与 cos/sin：对每个位置 $m$，$\theta = m \cdot \text{inv\_freq}$，再沿最后一维复制拼成 $d$ 维
   $$ \text{emb} = [\theta \,\Vert\, \theta],\quad \cos = \cos(\text{emb}),\ \sin = \sin(\text{emb}) \quad (\text{形状 } (B,S,d)) $$
3. 施加旋转（`rotate_half` 记 $x=[x_1,x_2]$ 两半，$[x_1,x_2]\mapsto[-x_2,x_1]$）：
   $$ q_{rot} = q \odot \cos + \text{rotate\_half}(q) \odot \sin $$
   $k$ 同理。cos/sin 需在 head 维广播（即对 $(B,S,d)$ 在 head 轴上展开）。

> 判题环境装有 `transformers`：提交时会与 `Qwen2RotaryEmbedding` + `apply_rotary_pos_emb`
> （同 `position_ids`、同 $d$、同 $base$）对拍。cos/sin 用 **float32** 计算。

## 示例

```
d = 4, base = 100, position_ids = [[1]], q = [1, 2, 3, 4]
inv_freq = [100^0, 100^-0.5] = [1.0, 0.1]
θ = [1.0, 0.1] → emb = [1.0, 0.1, 1.0, 0.1]
q_rot ≈ [-1.984111, 1.590675, 2.462378, 4.179684]
```

## 约束

- $d$ 为偶数；$q$ 形状 $(B,H_q,S,d)$、$k$ 形状 $(B,H_k,S,d)$（GQA 下 $H_q \ne H_k$ 需各自正确广播）
- `position_ids` 形状 $(B,S)$，dtype 整数
- 返回 `(q_rot, k_rot)` 二元组，形状与输入一致、dtype float32
- `torch.allclose` 相对容差 $10^{-5}$ 对拍 `transformers` 参考实现；不修改输入
