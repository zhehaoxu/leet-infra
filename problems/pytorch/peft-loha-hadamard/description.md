# PEFT：LoHa Hadamard 积低秩适配

## 题目描述

**LoHa**（Hadamard Product for Low-Rank Adaptation，源自 FedPara 系列，Hyeon-Woo et al., arXiv:2108.04916；PEFT 库 `tuners/loha` 实现）是 LoRA 的「乘法孪生兄弟」：LoRA 把两个低秩因子以**加法**拼进权重（\(W' = W_0 + s\,BA\)），而 LoHa 让两条低秩分支做**逐元素乘（Hadamard 积）**：

\[\Delta W = s\,\bigl(W_{1a}W_{1b}\bigr)\odot\bigl(W_{2a}W_{2b}\bigr),\qquad s=\frac{\alpha}{r}\]

其中 \(W_{1a},W_{2a}\in\mathbb R^{d_\text{out}\times r}\)，\(W_{1b},W_{2b}\in\mathbb R^{r\times d_\text{in}}\)（与 PEFT 库 `hada_w1_a/hada_w1_b` 等参数的形状一致），\(\odot\) 为逐元素乘。乘性组合的意义：两个秩-\(r\) 因子的 Hadamard 积可以达到**满秩** \(d_\text{out}\times d_\text{in}\) 的更新，而参数量仍只有两份 \(r(d_\text{out}+d_\text{in})\)——FedPara 论文正是用这一点在同等参数量下逼近全量微调。

PEFT 库的 Linear 实现里（`make_weight`），\(\Delta W\) 先按上式重建，再合并回基座：\(W' = W_0 + \Delta W\)，推理就只剩一次矩阵乘。

## 任务

给定冻结基座权重 \(W_0\) 与 LoHa 的两组因子、缩放参数 \(\alpha\) 和秩 \(r\)，返回**合并后的权重矩阵** \(W' = W_0 + s\,(W_{1a}W_{1b})\odot(W_{2a}W_{2b})\)，其中 \(s=\alpha/r\)。

- 输入 \(W_0\in\mathbb R^{d_\text{out}\times d_\text{in}}\)，\(W_{1a},W_{2a}\in\mathbb R^{d_\text{out}\times r}\)，\(W_{1b},W_{2b}\in\mathbb R^{r\times d_\text{in}}\)，\(\alpha\) 为实数、\(r\) 为正整数。
- 输出 \(W'\in\mathbb R^{d_\text{out}\times d_\text{in}}\)。

**约定**：不使用任何现成 PEFT/LyCORIS 库，直接用张量运算实现；结果按 float32 语义判定（相对/绝对容差 \(10^{-5}\)）。

## 提示

- 先各自算两条分支的 \(W_{ia}W_{ib}\)，再逐元素相乘，最后乘 \(s\) 加到 \(W_0\) 上。
- 乘法有一个重要退化：**任一分支的因子全零，\(\Delta W\) 就是全零**——PEFT/LyCORIS 的默认初始化正是把 \(W_{2b}\) 置零，让适配器在训练起点"什么都不做"。
