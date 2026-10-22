# 多头注意力前向（Multi-Head Attention）

## 题目描述

多头注意力把特征维 $D$ 拆成 $H$ 个头，每头在 $d_h = D/H$ 维子空间独立做缩放点积注意力，再把各头输出拼接回 $D$。

$$ \text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1,\dots,\text{head}_H),\quad \text{head}_i = \text{Attention}(Q_i,K_i,V_i) $$

## 规则

- 实现 `solve(q, k, v, num_heads, mask=None)`，其中 `q/k/v` 形状为 `(B, L, D)`，返回 `(B, L_q, D)`。
- `D` 必须能被 `num_heads` 整除。
- **维度顺序陷阱**：头是按最后一维的**连续块**切分的。拆头用 `view(B, L, H, dh).transpose(1, 2)` 得到 `(B, H, L, dh)`；合头用 `transpose(1, 2).reshape(B, L, D)`，输出列顺序必须是 `[head0 | head1 | ...]`，交换头顺序会答错。
- 每头按 $1/\sqrt{d_h}$ 缩放（注意是子头维度 $d_h$，不是 $D$）。
- mask 语义与缩放点积注意力一致（bool：`True=可见`，`False=屏蔽`），形状 `(..., L_q, L_k)` 或 `None`。

## 示例

```
输入：q=ones(1,2,4), k=ones(1,2,4), num_heads=2 (dh=2)
      v=[[[0,1,10,20],[2,3,30,40]]]
q、k 全 1 → 每头分数相同 → 权重均匀(各0.5) → 每头输出=该头对应 v 列块的均值
输出：[[[1,2,20,30],[1,2,20,30]]]   # head0=mean(v[...,0:2]), head1=mean(v[...,2:4])
```

## 约束

- float32/float64 输入
- 与逐头循环调用单头注意力（同 `F.scaled_dot_product_attention`）结果一致
