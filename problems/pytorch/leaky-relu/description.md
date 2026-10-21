# Leaky ReLU

## 题目描述

ReLU 在负区梯度恒为 0（「死神经元」）。Leaky ReLU 给负区一个小斜率保留梯度：

$$
\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} \cdot \min(0, x) =
\begin{cases} x, & x \ge 0 \\ \text{negative\_slope} \cdot x, & x < 0 \end{cases}
$$

## 规则

- 实现 `solve(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor`。
- 输出形状、dtype 与输入一致；`negative_slope=1` 时退化为恒等，`=0` 时退化为 ReLU。
- 要求向量化实现（如 `torch.where`），不能用 Python 循环。

## 示例

```
输入：x = [-2.0, -1.0, 0.0, 1.0, 2.0], negative_slope = 0.01
输出：[-0.02, -0.01, 0.0, 1.0, 2.0]
```

## 约束

- 输入为 float32 或 float64 张量，任意形状
- `negative_slope` 为非负浮点标量
