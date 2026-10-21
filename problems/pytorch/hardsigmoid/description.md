# HardSigmoid

## 题目描述

HardSigmoid 用一段截断线性函数近似 Sigmoid，输出恒落在 $[0,1]$：

$$
\text{HardSigmoid}(x) = \frac{\text{clip}(x+3,\,0,\,6)}{6} =
\begin{cases} 0, & x \le -3 \\ (x+3)/6, & -3 < x < 3 \\ 1, & x \ge 3 \end{cases}
$$

它是 HardSwish 的系数部分（HardSwish $= x \cdot$ HardSigmoid）。

## 规则

- 实现 `solve(x: torch.Tensor) -> torch.Tensor`，形状、dtype 与输入一致。
- 向量化实现；输出必须落在 $[0,1]$。

## 示例

```
输入：[-4.0, -3.0, 0.0, 3.0, 4.0]
输出：[0.0, 0.0, 0.5, 1.0, 1.0]
```

## 约束

- 输入为 float32 或 float64 张量，任意形状
- 斜坡段 $x=-1$ 处输出 $(-1+3)/6 = 1/3$
