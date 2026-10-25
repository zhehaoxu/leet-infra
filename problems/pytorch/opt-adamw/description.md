# 优化器：AdamW 解耦权重衰减（Decoupled Weight Decay）

## 题目描述

Adam 把 \(L_2\) 正则直接加进梯度，于是正则项也被自适应分母缩放，导致权重衰减实际强度随梯度尺度漂移。Loshchilov & Hutter（arXiv:1711.05101）提出 **AdamW**：把权重衰减从梯度里**解耦**出来，直接、按比例地收缩参数本身，\(L_2\) 正则与自适应学习率各管各的。它是当代大模型训练的默认优化器。

本题要求你**继承 PyTorch 的 `torch.optim.Optimizer` 基类**、实现一个 AdamW，并通过它的 `step()` 逐参数更新——而不是写一个纯函数。评测会构造若干待优化张量交给你的优化器，反复喂入固定梯度序列并调用 `step()`，比对最终参数与参考实现（原生 `torch.optim.AdamW`，`amsgrad=False`）是否逐元素一致。

## 规则

- 实现工厂函数 `solve(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)`，返回一个 `torch.optim.Optimizer` 子类的实例；`params` 是待优化张量（普通 `torch.Tensor` 列表）。实例的 `step()` 必须**就地更新** `params` 里的这些张量本身（评测靠读取原始张量取值判定）。
- 更新式严格复刻 `torch.optim.AdamW`（`amsgrad=False`、无 `maximize`）。对每个在第 \(t\) 步（\(t\) 从 1 计）有梯度的参数 \(p\)、梯度 \(g\)：
  1. **解耦权重衰减**（仅当 `weight_decay`≠0，且发生在矩估计之前）：\(p \leftarrow p\,(1 - lr\cdot\text{wd})\)
  2. 一阶矩：\(m \leftarrow \beta_1 m + (1-\beta_1) g\)
  3. 二阶矩：\(v \leftarrow \beta_2 v + (1-\beta_2)\, g^2\)
  4. 偏差修正：\(\hat m = \dfrac{m}{1-\beta_1^{\,t}},\quad \hat v = \dfrac{v}{1-\beta_2^{\,t}}\)
  5. 参数更新：\(p \leftarrow p - \dfrac{lr}{1-\beta_1^{\,t}}\cdot\dfrac{m}{\sqrt{v}/\sqrt{1-\beta_2^{\,t}} + \epsilon}\)
- `grad is None` 的参数必须原样跳过（连权重衰减也不施加）。
- 计算全程 float32；每个参数的 `step` 计数独立维护。**禁止**直接实例化或调用 `torch.optim.AdamW` / `torch.optim.Adam`，需手写更新式。

## 示例

初始化单个参数 \(p=[1.0,\,-2.0,\,0.5]\)，`lr=0.1`、`betas=(0.9, 0.999)`、`eps=1e-8`、`weight_decay=0.01`，连续喂入三步梯度：

```
g₁ = [ 0.10, -0.20,  0.30]
g₂ = [-0.40,  0.05,  0.20]
g₃ = [ 0.15, -0.10, -0.25]
三步后 p ≈ [0.971888, -1.786759, 0.276560]
```

## 约束

- 输入张量 dtype 为 float32；输出就地更新、dtype 不变
- 支持 1D（偏置/归一化增益）与多维（权重矩阵）混合的参数列表
- 须与原生 `torch.optim.AdamW` 逐步完全一致（对拍口径见 notes.md）
