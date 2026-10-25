# 优化器：Adafactor 因子分解二阶矩（Factored Second Moment）

## 题目描述

Adam 要为每个参数维护一份与参数同形状的二阶矩 \(v\)，显存翻倍。Adafactor（Shazeer & Stern, arXiv:1804.04235）对矩阵参数的 \(v\) 做**行、列两个低秩统计量的外积近似**，把二阶矩状态从 \(O(rc)\) 降到 \(O(r+c)\)；同时用 **RMS 更新裁剪**替代固定学习率、并可选地按参数 RMS 缩放步长。它是 T5 等模型的默认优化器之一。

本题要求你**继承 `torch.optim.Optimizer`** 实现 Adafactor（本文采用其 BigVision 变体的常见约定），通过 `step()` 就地更新参数；评测喂固定梯度序列、比对最终参数与独立参照实现是否逐元素一致。为在 CPU 上稳定判题，全部计算用 **float32**（不做 bfloat16 动量、不启用 `relative_step`）。

## 规则

实现工厂函数 `solve(params, lr=1e-2, betas=(0.9, 0.999), eps1=1e-30, eps2=1e-3, decay_rate=-0.8, clip_threshold=1.0, scale_parameter=True, weight_decay=0.0)`，返回 `torch.optim.Optimizer` 子类实例；`step()` 须就地更新 `params`。

设组步计数 \(t\) 从 1 起、\(\beta_1,\beta_2^{cap}\) 来自 `betas`。对每个有梯度 \(g\) 的参数 \(p\)：

1. 逐步二阶衰减：\(\beta_{2,t} = \min\!\big(\beta_2^{cap},\; 1 - t^{\text{decay\_rate}}\big)\)，令 \(b = 1 - \beta_{2,t}\)。
2. 有效学习率：若 `scale_parameter`，\(lr_{eff} = \max(\epsilon_2, \operatorname{RMS}(p))\cdot lr\)，否则 \(lr_{eff}=lr\)；其中 \(\operatorname{RMS}(x)=\sqrt{\frac{1}{n}\sum_i x_i^2}\)。
3. 梯方：\(U = g^2 + \epsilon_1\)。
4. 二阶矩：
   - **矩阵参数（\(g\) 的维数 \(\ge 2\)，取末两维为行 \(r\)、列 \(c\)）**：行/列统计量各自 EMA
     \[ v_r \leftarrow (1-b)v_r + b\cdot \operatorname{rowmean}(U),\quad v_c \leftarrow (1-b)v_c + b\cdot \operatorname{colmean}(U) \]
     重构 \(\hat v_{ij} = \dfrac{v_{r,i}}{\sum_k v_{r,k}}\cdot v_{c,j}\)。
   - **非标量维数 \(<2\)**：整块 EMA \(v \leftarrow (1-b)v + b\,U\)，\(\hat v = v\)。
5. 预条件方向：\(u = \big(\hat v^{-1/2}\big)\odot g\)。
6. **RMS 更新裁剪**：\(r_u=\operatorname{RMS}(u)\)，\(\text{factor}=\max(r_u/\text{clip\_threshold},\,1)\)，\(u \leftarrow u/\text{factor}\cdot lr_{eff}\)。
7. 一阶动量：\(m \leftarrow \beta_1 m + (1-\beta_1) u\)，取 \(u \leftarrow m\)。
8. 解耦权重衰减（`weight_decay`≠0 时）：\(p \leftarrow p\,(1 - lr_{eff}\cdot\text{wd})\)。
9. 更新：\(p \leftarrow p - u\)。

初始状态全零；`grad is None` 的参数跳过。**禁止**调用现成 Adafactor/Adam，需手写。

## 示例

矩阵参数 \(p=\begin{bmatrix}0.5 & -0.2\\ 0.3 & 0.4\\ -0.1 & 0.2\end{bmatrix}\)，`lr=0.1`、`betas=(0.9,0.999)`、`scale_parameter=True`、`weight_decay=0`，三步梯度：

```
g₁ = [[ 0.2,-0.1],[ 0.1, 0.3],[-0.2, 0.05]]
g₂ = [[-0.1, 0.25],[ 0.2,-0.15],[ 0.05, 0.1]]
g₃ = [[ 0.15,-0.05],[-0.2, 0.1],[ 0.1,-0.3]]
三步后 p ≈ [[0.489838, -0.202574], [0.292380, 0.392157], [-0.093338, 0.196794]]
```

## 约束

- 输入 float32；就地更新、dtype 不变
- 行/列均按末两维统计；因子化路径针对**矩阵参数（维数恰为 2）**，\(\sum_k v_{r,k}\) 为对整条行统计向量求和（首步 \(b=1\)、梯方为正，故分母非零）；维数 1/0 走整块 EMA
- 与独立参照逐步完全一致（约定与口径见 notes.md）
