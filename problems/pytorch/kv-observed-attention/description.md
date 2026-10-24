# ObservedAttention：历史注意力平均分（H2O 家族）

## 题目描述

**ObservedAttention**（与 H2O [arXiv:2306.14048](https://arxiv.org/abs/2306.14048) 同源）：一个 key 越重要，历史 query 打在它上面的注意力就越多。分数就是 **每个 key 收到的注意力之和 ÷ 能看到它的 query 数**——即"过去每一步平均给了它多少注意力"。

kvpress 里 `ObservedAttentionPress.score` 三步：
1. `scores = attentions.sum(2)` — 对 query 维求和 → 每 key 收到的 attention 累计
2. `scores /= arange(L, 0, -1)` — 位置 $j$ 只被 $\geq j$ 的 $L - j$ 个 query 看到（因果），除以 $L - j$ 得平均
3. `view(H_kv, groups, L).mean(1)` — 把 num_heads 摊到 num_kv_heads 组内求平均

## 规则

给定 `attn` 形状 $(H_q, L, L)$（**已 softmax 且因果下三角**：`attn[i, j] = 0` 若 `j > i`），`num_kv_heads = H`：返回 shape $(H, L)$ 分数。

- `groups = H_q // H`
- 对每 head $h$ 与位置 $j$：
$$ \text{score}_{h,j} = \frac{1}{\text{groups}} \sum_{g=0}^{\text{groups}-1} \sum_{i=j}^{L-1} \text{attn}_{h\cdot \text{groups}+g,\, i,\, j} \;\Big/\; (L - j) $$

## 示例

```
Hq=2, L=4, num_kv_heads=1, groups=2

head0 attn[0]:          head1 attn[1]:
[1.0, 0.0, 0.0, 0.0]    [1.0, 0.0, 0.0, 0.0]
[0.5, 0.5, 0.0, 0.0]    [0.4, 0.6, 0.0, 0.0]
[0.2, 0.3, 0.5, 0.0]    [0.3, 0.3, 0.4, 0.0]
[0.1, 0.2, 0.3, 0.4]    [0.2, 0.1, 0.3, 0.3]

对每 key 列 j，sum 沿 query i 维：
  head0 累计 = [1.8, 1.0, 0.8, 0.4]
  head1 累计 = [1.9, 1.0, 0.7, 0.3]
除以 (L-j) = [4, 3, 2, 1]：
  head0 平均 = [0.45, 0.333, 0.40, 0.40]
  head1 平均 = [0.475, 0.333, 0.35, 0.30]
kv-group mean = 平均两 head：
  scores = [[0.4625, 0.3333, 0.375, 0.35]]
```

## 约束

- CPU float32；`H_q` 必须能被 `H` 整除（题目输入保证）。
- 因果掩码已由上游施加，越界位就是 0，不再特殊处理。
