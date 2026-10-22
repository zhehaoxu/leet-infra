# 缩放点积注意力（Scaled Dot-Product Attention）

## 题目描述

缩放点积注意力是 Transformer 的基本单元：用 Query 与 Key 的相似度经 softmax 得到权重，再对 Value 加权求和。

$$ \text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + \text{mask}\right)V $$

其中 $d_k$ 是 Key 的最后一维（特征维），$\sqrt{d_k}$ 缩放用于抑制内积随维度增大而方差膨胀。

## 规则

- 实现 `solve(query, key, value, mask=None)`，返回与 query 批次结构一致、特征维等于 `value.shape[-1]` 的张量。
- 输入支持 `(..., L, d)` 的任意前置批量维（batch、head 等），逐行做 softmax。
- **mask 采用 torch 原生 bool 语义**（与 `F.scaled_dot_product_attention` 一致）：`True = 允许注意力`，`False = 屏蔽`（在屏蔽位置置 $-\infty$，softmax 后概率为 0）。若某一行被整行屏蔽，则该行输出为 0（torch 约定，不得产生 NaN）。
- softmax 必须沿 key 方向（倒数第二维对应的最后一轴）做，并对每行减去最大值以保证数值稳定（大内积不减 max 会上溢）。

## 示例

```
输入：query=[[[1,0]]], key=[[[1,0],[1,0]]], value=[[[3,0],[1,0]]]
两个 key 相同 → 分数相等 → 权重各 0.5
输出：[[[2.0, 0.0]]]        # 0.5*3 + 0.5*1 = 2.0
```

## 约束

- float32/float64 输入
- 结果与 `torch.nn.functional.scaled_dot_product_attention` 一致（bool mask 且非全屏蔽行）
