# im2col 加速卷积

## 题目描述

im2col 是把卷积「摊平」成矩阵乘的经典工程技巧：将每个输出位置的感受野展开拉平成一行，得到列矩阵 $A \in \mathbb{R}^{(N \cdot o_h o_w) \times (C_{in} k_h k_w)}$；把权重 reshape 成 $W \in \mathbb{R}^{(C_{in} k_h k_w) \times C_{out}}$，则

$$
\text{out}_{\text{flat}} = A \cdot W
$$

一次 GEMM 就完成了原本几万次的循环。它是 cuDNN 早期版本卷积的骨架，也是理解「为什么 GEMM kernel 优化等于卷积优化」的钥匙。

## 规则

- 实现 `solve(x, weight, bias=None, stride=1, padding=0)`，形状约定同「二维卷积前向」：x $(N, C_{in}, H, W)$，weight $(C_{out}, C_{in}, k_h, k_w)$。
- 流程：0-padding → 用 `unfold`（或循环）收集每个感受野成 $(N, o_h o_w, C_{in} k_h k_w)$ → 与 reshape 后的权重做一次矩阵乘 → reshape/permute 回 $(N, C_{out}, o_h, o_w)$。
- 矩阵乘只允许在最后一步用 `@`/`matmul`（这正是本题的语义：卷积 → GEMM）；感受野的收集必须手写。
- **陷阱**：展开行的元素顺序必须是「通道 → 行 → 列」（与 weight 的 $(C_{in}, k_h, k_w)$ 展平顺序一致），两个矩阵按同一个顺序摊平才能对上。
- 禁止调用 `F.conv2d`。

## 示例

```
输入：x = 单通道 3x3 = [[1,2,3],[4,5,6],[7,8,9]], weight = 1个 2x2 全 1 核
col 矩阵（4 个 2x2 感受野拉平）：[[1,2,4,5],[2,3,5,6],[4,5,7,8],[5,6,8,9]]
输出：col @ ones(4x1) = [[12,16,24,28]] → [[[12,16],[24,28]]]
```

## 约束

- float32/float64；测试尺寸较小（im2col 内存膨胀 $k_h k_w C_{in}$ 倍可接受）
- 结果与 `torch.nn.functional.conv2d` 一致
