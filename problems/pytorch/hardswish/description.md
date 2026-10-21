# HardSwish

## 题目描述

HardSwish 是 Swish 的分段线性近似，去掉 expensive 的 sigmoid，用截断线性函数替代，移动端常用：

$$
\text{HardSwish}(x) = x \cdot \frac{\text{clip}(x+3,\,0,\,6)}{6} =
\begin{cases} 0, & x \le -3 \\ x(x+3)/6, & -3 < x < 3 \\ x, & x \ge 3 \end{cases}
$$

## 规则

- 实现 `solve(x: torch.Tensor) -> torch.Tensor`，形状、dtype 与输入一致。
- 向量化实现（`clamp`/`clip` 到 $[0,6]$ 后乘 $x/6$）。

## 示例

```
输入：[-4.0, -3.0, 0.0, 3.0, 4.0]
输出：[0.0, 0.0, 0.0, 3.0, 4.0]
```

## 约束

- 输入为 float32 或 float64 张量，任意形状
- $x \le -3$ 输出 0，$x \ge 3$ 输出 $x$（两侧平台区）
