# FlashAttention 分块在线 softmax

## 题目描述

FlashAttention 的核心思想是**不显式构造完整的 $L_q \times L_k$ 注意力矩阵**，而是把 query 分块，
沿 key 维一块一块地累加，用「在线 softmax」（running max + running sum）保证数值稳定，
最终得到与标准注意力**数值等价**的输出。本题要求你手写这套分块累加逻辑。

## 规则

- 实现 `solve(q, k, v, block)`，单头注意力：`q` 形状 `(B, Lq, d)`，`k`、`v` 形状 `(B, Lk, d)`，返回 `(B, Lq, d)`。
- 缩放为 $1/\sqrt{d}$；无掩码。
- 必须以 `block` 为大小对 query、key 两个维度**分块遍历**，用在线归约维护每行的
  最大值 $m$、分母累量 $l$、分子累量 $\text{acc}$，并在遇到更大的 $m$ 时按
  $\alpha=\exp(m_{old}-m_{new})$ 对已累加的 $l,\text{acc}$ 做 rescale。
- 结果需与标准 $\text{softmax}(QK^\top/\sqrt{d})V$ 一致，且对**任意 `block` 取值**、
  以及 `block` 不整除 $L_q/L_k$、$L_k \neq L_q$ 都成立。

## 示例

```
输入：q=[[[1,0],[0,1],[1,1]]]
     k=[[[1,0],[1,0],[0,1]]]
     v=[[[1,2],[3,4],[5,6]]]
     block=2
输出：[[[2.593328, 3.593328],
        [3.510469, 4.510469],
        [3.000000, 4.000000]]]
（与标准注意力结果一致，仅浮点末位有差异）
```

## 约束

- float32 输入；$B \ge 1$，$L_q, L_k, d \ge 1$，`block >= 1`
- logits 可能很大（如 $\times 20$），在线 max 归约须保证其数值稳定、不溢出
