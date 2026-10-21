# PReLU（可学习负斜率）

## 题目描述

PReLU 把 Leaky ReLU 的负斜率变成一个**可学习参数**，且通常逐通道（per-channel）共享：

$$
\text{PReLU}(x) =
\begin{cases} x, & x > 0 \\ w \cdot x, & x \le 0 \end{cases}
$$

其中 $w$ 是与通道数对齐的张量。

## 规则

- 实现 `solve(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor`。
- `weight` 为标量张量，或长度为**通道数 C** 的一维张量（按 PyTorch 约定作用于 `dim=1` 通道维，需 reshape 成 `(1, C, 1, 1, ...)` 再广播）。
- 输出形状、dtype 与 `x` 一致；向量化实现。
- `weight=0` 时退化为 ReLU。

## 示例

```
输入：x = [-2.0, -1.0, 0.0, 1.0, 2.0], weight = 0.25（标量）
输出：[-0.5, -0.25, 0.0, 1.0, 2.0]
```

逐通道示例：`x = [[-2, -4, 3], [-1, -5, 6]]`，`weight = [0.1, 0.5, 0.9]`（通道维长度 3）
→ 输出负元素按各自通道乘：`[[-0.2, -2.0, 3.0], [-0.1, -2.5, 6.0]]`

## 约束

- `x` 为 float32/float64，形状 `(N, C)` 或 `(N, C, ...)`
- 一维 `weight` 的长度必须等于 `x` 在 `dim=1` 的通道数
