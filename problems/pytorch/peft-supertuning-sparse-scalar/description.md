# PEFT：SuperTuning 稀疏标量支撑与 LoRA 混合

## 题目描述

**SuperTuning**（arXiv:2607.09287）在基座权重 \(W_0\) 的**幅值最大位置**上放置可训练的稀疏标量，形成对基座的精细局部修正；同时支持与 LoRA 的混合变体 **Supra**（SuperTuning + LoRA）：

\[\Delta W = M_k(W_0)\odot S + \mathbf 1_{r>0}\,\frac{\alpha}{r}\,BA,\]

其中 \(M_k(W_0)\) 是 \(W_0\) 按绝对值取 top-\(k\) 的 0/1 掩码（并列时按下标小者优先），\(S\in\mathbb R^{d_\text{out}\times d_\text{in}}\) 是可训练标量更新矩阵，\(BA\) 是可选 LoRA 旁路。当 \(r=0\) 时只保留稀疏标量项；当同时给定 LoRA 且 \(\alpha>0\) 时二者相加。

## 任务

给定 \(W_0\in\mathbb R^{d_\text{out}\times d_\text{in}}\)、稀疏更新 \(S\)、掩码预算 \(k\ge 0\)、可选 LoRA 因子 \(A\in\mathbb R^{r\times d_\text{in}},B\in\mathbb R^{d_\text{out}\times r}\) 以及 \(\alpha,r\)，返回合并权重：

\[W' = W_0 + M_k(W_0)\odot S + \mathbf 1_{r>0}\,\frac{\alpha}{r}\,BA.\]

- \(k=0\) 时掩码全零，只保留 LoRA 项（若 \(r>0\)）。
- \(r=0\) 时忽略 \(A,B,\alpha\)，只保留稀疏标量项。
- 对 \(|W_0|\) 做**稳定降序排序**，并列元素按下标小者优先进入 top-\(k\)。
- 输出 \(W'\in\mathbb R^{d_\text{out}\times d_\text{in}}\)，float32。

## 提示

- 先把 \(W_0\) 拉平求绝对值，稳定排序取前 \(k\) 个下标，再 reshape 回矩阵形状得到 \(M_k\)。
- 当 \(\alpha=0\) 或 \(S=0\) 时，相应项消失；注意 \(M_k\) 只控制 \(S\) 的稀疏支撑，不影响 \(W_0\) 本身。
- 本题中 \(A/B\) 的命名与 `peft-lora-forward` 一致：\(A\in\mathbb R^{r\times d_\text{in}},B\in\mathbb R^{d_\text{out}\times r}\)。
