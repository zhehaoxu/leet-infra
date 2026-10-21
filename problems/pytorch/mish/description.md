# Mish

## 题目描述

Mish 是一条自门控的平滑曲线，负区有轻微下凹（类似 Swish），论文称在某些任务上优于 ReLU：

$$
\text{Mish}(x) = x \cdot \tanh\!\big(\text{SoftPlus}(x)\big) = x \cdot \tanh\!\left(\log(1+e^{x})\right)
$$

## 规则

- 实现 `solve(x: torch.Tensor) -> torch.Tensor`，形状、dtype 与输入一致。
- 允许使用 `torch.tanh`、`torch.exp`、`torch.log1p` 等原语。
- **数值稳定**：内部的 $\text{SoftPlus}(x)=\log(1+e^x)$ 必须按符号分支稳定计算（`exp(x)` 大值上溢、$x$ 接近 0 时精度），否则 Mish 会得 NaN。推荐 `softplus = where(x>0, x + log1p(exp(-x)), log1p(exp(x)))`。
- 向量化实现。

## 示例

```
输入：[-1.0, 0.0, 1.0]
输出：[-0.30340147, 0.0, 0.86509842]
```

## 约束

- 输入为 float32 或 float64 张量，任意形状
- $x$ 很大时输出 $\to x$，$x$ 很负时输出 $\to 0$（都不得出现 NaN/inf）
