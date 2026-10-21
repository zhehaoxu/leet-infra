# SoftPlus

## 题目描述

SoftPlus 是 ReLU 的平滑版本，也是 GELU/Mish 等激活的构件：

$$
\text{SoftPlus}(x) = \frac{1}{\beta}\log\!\left(1 + e^{\beta x}\right)
$$

当 $\beta x$ 很大时它逼近 $x$；当 $\beta x$ 很小时逼近 $\frac{1}{\beta}e^{\beta x}$（趋 0）。

## 规则

- 实现 `solve(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor`。
- **数值稳定**：`exp(beta*x)` 在 `beta*x` 很大时上溢。按 PyTorch 约定，当 `beta*x > threshold` 时直接返回 `x`（此区 softplus ≈ x），其余用 `log1p(exp(beta*x))/beta`。用 `log1p` 而非 `log(1+·)` 保近零精度。
- 向量化实现。

## 示例

```
输入：x = [-2.0, 0.0, 2.0], beta = 1.0, threshold = 20.0
输出：[0.12692802, 0.69314718, 2.12692809]     # softplus(0)=ln2≈0.6931
```

## 约束

- 输入为 float32 或 float64 张量，任意形状
- `beta > 0`；可能包含 `beta*x` 达数百的大值，需走线性分支不溢出
