# MSE 均方误差损失（MSE Loss）

## 题目描述

均方误差（Mean Squared Error）是回归任务最常用的损失：预测值与真值之差的平方的平均。它惩罚大误差（误差翻倍、损失四倍），且处处可导。

## 规则

- 实现 `solve(pred, target, reduction="mean")`，逐元素先算 $(\text{pred}-\text{target})^2$，再按 `reduction` 归约：
  - `"mean"`：所有元素求平均（默认）；
  - `"sum"`：所有元素求和；
  - `"none"`：不归约，返回与 `pred` 同形的逐元素损失。
- `pred` 与 `target` 形状相同（或可广播）。**注意**：MSE 是 $\frac1n\sum(p_i-t_i)^2$，前面**没有** $\frac12$ 系数。
- 未知 `reduction` 抛 `ValueError`。
- 结果须与 `F.mse_loss(pred, target, reduction=...)` 一致。

## 示例

```
输入：pred = [2.5, 0.0, 2.0, 8.0]
      target = [3.0, -1.0, 2.0, 7.0]
reduction="none" → [0.25, 1.0, 0.0, 1.0]
reduction="mean" → 0.5625
reduction="sum"  → 2.25
```

## 约束

- 支持 float32 / float64；`reduction` 为 `"none"` 时输出 dtype 与 `pred` 一致
- 标量归约（mean/sum）返回 0 维张量，与 `F.mse_loss` 行为一致
