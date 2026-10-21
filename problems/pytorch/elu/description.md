# ELU

## 题目描述

ELU（Exponential Linear Unit）负区用指数饱和到 $-\alpha$，均值更接近 0、收敛更快：

$$
\text{ELU}(x) =
\begin{cases} x, & x > 0 \\ \alpha\,(e^{x} - 1), & x \le 0 \end{cases}
$$

## 规则

- 实现 `solve(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor`，形状、dtype 与输入一致。
- 向量化实现，不能用 Python 循环。
- **精度提示**：负区 $e^x - 1$ 在 $x$ 接近 0 时会有浮点抵消误差，用 `torch.expm1(x)` 直接算 $e^x-1$ 更准。

## 示例

```
输入：x = [-1.0, 0.0, 1.0], alpha = 1.0
输出：[-0.63212055, 0.0, 1.0]     # expm1(-1) = e^-1 - 1
```

## 约束

- 输入为 float32 或 float64 张量，任意形状
- `alpha` 为非负浮点标量；$x$ 很负时输出趋近 $-\alpha$
