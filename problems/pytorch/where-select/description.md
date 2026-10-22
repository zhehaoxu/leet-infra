# Where 条件选择（广播三目）

## 题目描述

`torch.where(condition, x, y)` 是元素级三目运算：`condition` 为 `True` 处取 $x$，否则取 $y$。三者可以形状不同，但必须能**广播**到同一形状。

$$ \text{out}[i] = \begin{cases} x[i] & \text{condition}[i] = \text{True} \\ y[i] & \text{否则} \end{cases} $$

## 规则

- 实现 `solve(condition, x, y)`，`condition` 为布尔（或按真值解释）张量，`x`/`y` 可为张量或标量。
- `condition`、`x`、`y` 按**标准广播规则**对齐到公共形状后逐元素选择。
- dtype 遵循 `x`/`y` 的类型提升（整数与浮点混合输出浮点）。
- 与 `torch.where` 一致。

## 示例

```
输入：condition=[True,False,True,False], x=[1,2,3,4], y=[-1,-2,-3,-4]
输出：[1, -2, 3, -4]      # condition 真取 x、假取 y
```

## 约束

- 可广播的任意形状；广播失败时应报错
