# TOVA：最后一个 query 的注意力跨头平均

## 题目描述

**TOVA**（Otsuka et al., 2024, [arXiv:2401.06104](https://arxiv.org/abs/2401.06104)）：只用**最后一个 query 对更早 key 的注意力**、跨 head 求平均——每个 key 的重要性就是"最后一步模型最关心它多少"。

kvpress 里 `TOVAPress.score` 三步：
1. `attn_weights = attentions[..., -1:, :-1]` — 最后一行 query、去掉最后一个 key（自注意力位）
2. `scores = attn_weights.mean(1)` — 跨 num_heads 平均（dim=1 在有 batch 时是 head）
3. `scores = repeat(num_kv_heads, 1)` + 尾部 pad `max+1`（保底最后一个 token）

## 规则

给定 `attn` 形状 $(H_q, L, L)$（**每行 softmax 后和为 1**、因果下三角已生效），`num_kv_heads`（GQA 里的 KV 头数 $H$）：返回 shape $(H, L)$ 的分数：

1. 取 `attn[:, L-1, :L-1]` → $(H_q, L-1)$
2. 跨 $H_q$ 维取 mean → $(L-1,)$（**所有 head 一个共享分数**）
3. 复制到 $H$ 行 → $(H, L-1)$
4. 尾部 pad 1 位，值 = `全局 max + 1` → $(H, L)$

## 示例

```
Hq=4, L=5, num_kv_heads=2
每 head 最后一行 attn[:, 4, :4]：
  h=0: [0.1, 0.2, 0.3, 0.3]
  h=1: [0.2, 0.2, 0.2, 0.3]
  h=2: [0.15, 0.15, 0.3, 0.3]
  h=3: [0.1, 0.3, 0.2, 0.3]
mean over heads = [0.1375, 0.2125, 0.25, 0.3]
max+1 = 1.3
scores = [[0.1375, 0.2125, 0.25, 0.3, 1.3],
          [0.1375, 0.2125, 0.25, 0.3, 1.3]]
```

## 约束

- CPU float32；`attn` 已经是 softmax + 因果掩码后的权重。
- $H_q$ 不必等于 $H$；分数**先跨 $H_q$ 平均**、再复制 $H$ 份。
