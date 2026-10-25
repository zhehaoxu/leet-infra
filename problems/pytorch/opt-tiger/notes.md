# 详解：Tiger 动量符号更新

## 出处与说明
参考实现为 kozistr/pytorch-optimizer 的 `Tiger`（`weight_decouple=True`、`maximize=False`），其文档描述为 “Sign based updates with a single gradient momentum buffer”。该算法属 **sign 型优化器家族**：与 Lion（arXiv:2305.14616）、SignSGD 同源思想——用更新方向的符号把每步幅度归一。Tiger 可视为“只有动量、且直接对动量取符号”的极简版本（Lion 是对 `β1·m+(1−β1)·g` 与 `g` 的插值取符号，Tiger 只对 EMA 动量取符号）。

> 说明：本合集要求来源可核。Tiger 的实现口径以 kozistr/pytorch-optimizer 为准（默认 β=0.965、解耦 wd=0.01）；未检索到独立权威论文，故不虚构 arXiv 编号，题目判据完全基于该实现与其更新式。

## 更新式与要点
每参数仅一份动量 \(m\)，初值 0：

\[ m \leftarrow \beta m + (1-\beta) g,\qquad p \leftarrow p - lr\cdot\operatorname{sign}(m) \]

- **无二阶矩、无偏差修正**：因为取了 \(\operatorname{sign}\)，更新幅度恒为 \(lr\)，与 \(m\) 的大小无关（只与符号有关）。这使得它对梯度尺度天然免疫，但也意味着早期动量还小时方向就已是 \(\pm 1\)。
- **符号函数在 0 处**：\(\operatorname{sign}(0)=0\)，该坐标该步不动。评测里随机浮点梯度几乎不会精确为 0，但用例保证 `grad is None` 的参数连权重衰减都跳过。
- **解耦权重衰减先于动量**：`p.mul_(1 - lr*wd)`，与 AdamW 同口径。

## 对拍口径
判题用独立重算（`_oracle_tiger`）：`m = beta*m+(1-beta)*g`、`p -= lr*torch.sign(m)`，多形状多步随机梯度下 `allclose(atol=1e-5, rtol=1e-5)` 比对。sign 是逐元素、离散判据，浮点噪声不改变符号（远离 0），故与解天然对齐。

## 复杂度
每参数每步 \(O(\text{numel})\)，额外状态仅一份动量 \(m\)，是 Adam 的一半。
