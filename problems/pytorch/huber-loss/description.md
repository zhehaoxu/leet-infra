# Huber 损失（Huber Loss / Smoothed Absolute Loss）

## 题目描述

Huber 损失是回归中的鲁棒损失：小残差时像 MSE（二次、平滑好优化），大残差时切成线性（像 MAE，不被离群点主导）。一个阈值 $\delta$ 控制切换点，兼顾收敛质量与抗噪能力。

## 规则

- 实现 `solve(pred, target, delta=1.0, reduction="mean")`，逐元素令 $z=|\text{pred}-\text{target}|$：
  $$ \mathcal L = \begin{cases}\tfrac12 z^2 & z\le\delta\\[3pt] \delta\,(z-\tfrac12\delta) & z>\delta\end{cases} $$
- 两段在 $z=\delta$ 处**函数值与一阶导都连续**（都是 $\tfrac12\delta^2$、斜率都是 $\delta$），这就是「smooth」。
- `reduction` 取 `"mean"` / `"sum"` / `"none"`；未知值抛 `ValueError`。
- 结果须与 `F.huber_loss(pred, target, delta=delta, reduction=...)` 一致。

## 示例

```
输入：pred = [1.5, -1.0, 0.3, 4.0]
      target = [1.0, 0.5, 0.2, 1.0]
      残差绝对值 z = [0.5, 1.5, 0.1, 3.0]
delta=1.0, reduction="none" → [0.125, 1.0, 0.005, 2.5]
delta=1.0, reduction="mean" → 0.9075
delta=0.5, reduction="mean" → 0.5325
（z>delta 的项走线性支：0.5 处 = delta*(z-0.5*delta)）
```

## 约束

- `pred` 支持 float32 / float64；`delta > 0`
- 不使用 `F.huber_loss` / `F.smooth_l1_loss`；需手写分段（用 `torch.where`）
