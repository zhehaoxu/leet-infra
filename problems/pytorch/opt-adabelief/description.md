# 优化器：AdaBelief 置信度二阶矩（Belief in Gradient Prediction）

## 题目描述

AdaBelief（arXiv:1904.03502）保留了 Adam 的一阶动量 \(m\)，但把二阶量从「梯度的平方」换成「**梯度对其动量预测的误差的平方**」\((g-m)^2\)。直观上，若梯度能被动量良好预测（误差小），则 \(s\) 小、分母小、学习率被放大；反之噪声大时误差大、分母大、步子自动收紧。论文称这能兼顾快速收敛与平坦极小值。

本题要求你**继承 `torch.optim.Optimizer`** 实现 AdaBelief（non-rectified 变体），通过 `step()` 就地更新参数；评测喂固定梯度序列、比对最终参数与独立参照是否逐元素一致。

## 规则

- 实现工厂函数 `solve(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16, weight_decay=0.0)`，返回一个 `torch.optim.Optimizer` 子类实例；`params` 是待优化张量列表，`step()` 须**就地更新**这些张量本身。
- 对每个第 \(t\) 步（\(t\) 从 1 计）有梯度的参数 \(p\)、梯度 \(g\)，状态 \(m, s\) 初始为零：
  1. **解耦权重衰减**（仅当 `weight_decay`≠0，先于方向计算）：\(p \leftarrow p\,(1 - lr\cdot\text{wd})\)
  2. 一阶动量：\(m \leftarrow \beta_1 m + (1-\beta_1)\, g\)
  3. **置信二阶矩**（用更新后的 \(m\)）：\(s \leftarrow \beta_2 s + (1-\beta_2)(g-m)^2 + \epsilon\)
  4. 偏差修正因子：\(\hat\beta_1^{(t)} = 1-\beta_1^t\)，\(\sqrt{\hat\beta_2^{(t)}}=\sqrt{1-\beta_2^t}\)
  5. 分母：\(\text{denom} = \sqrt{s}/\sqrt{\hat\beta_2^{(t)}} + \epsilon\)
  6. 更新：\(p \leftarrow p - \dfrac{lr}{\hat\beta_1^{(t)}}\cdot \dfrac{m}{\text{denom}}\)
- \(\epsilon\) 既加进 \(s\) 的累积、也加进最终分母（与库实现一致）。全程 float32。`grad is None` 的参数整体跳过（连 wd 也不施加）。**禁止**调用现成优化器，需手写。

## 示例

初始化单个参数 \(p=[1.0,\,-2.0,\,0.5]\)，`lr=0.1`、`betas=(0.9, 0.999)`、`eps=1e-16`、`weight_decay=0.01`，连续喂入三步梯度：

```
g₁ = [ 0.10, -0.20,  0.30]
g₂ = [-0.40,  0.05,  0.20]
g₃ = [ 0.15, -0.10, -0.25]
三步后 p ≈ [0.966739, -1.765123, 0.247988]
```

## 约束

- 输入张量 dtype 为 float32；输出就地更新、dtype 不变
- 支持任意形状参数；每个参数各维护一份 \((m, s)\) 状态
- 与独立参照逐步完全一致（口径见 notes.md）
