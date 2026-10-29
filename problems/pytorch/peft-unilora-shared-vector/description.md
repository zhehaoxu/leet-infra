# PEFT：UniLoRA 共享参数向量的索引分配

## 题目描述

**UniLoRA**（Unified Low-Rank Adapter，arXiv:2506.00799）把所有 LoRA 适配器的可训练参数压缩成**一条全局共享向量** \(\theta\in\mathbb R^D\)。每个线性层不再拥有自己独立的 \(A/B\)，而是用一个由 `proj_seed` 决定的**固定伪随机索引分配**，从 \(\theta\) 中取出本层所需的 \(r(d_\text{in}+d_\text{out})\) 个元素，分别 reshape 成 \(A\in\mathbb R^{r\times d_\text{in}}\) 与 \(B\in\mathbb R^{d_\text{out}\times r}\)，再按标准 LoRA 缩放合并：

\[W' = W_0 + \frac{\alpha}{r}\,B A.\]

`layer_id` 用于区分模型中不同层：每层的随机种子取 `proj_seed + layer_id`，从而保证同一模型内各层的索引分配互不重叠且完全确定。

## 任务

给定冻结基座 \(W_0\in\mathbb R^{d_\text{out}\times d_\text{in}}\)、共享向量 \(\theta\in\mathbb R^D\)、秩 \(r\)、缩放 \(\alpha\)、投影种子 `proj_seed` 与层号 `layer_id`，按上述规则返回合并后的 \(W'\)。

- 若 \(r(d_\text{in}+d_\text{out}) > D\)，可假设输入合法（测试不会触发）。
- 索引分配使用 `torch.randperm(D, generator=...)`，生成器种子设为 `int(proj_seed) + int(layer_id)`。
- 取出的前 \(r\cdot d_\text{in}\) 个元素 reshape 为 \(A\)（行优先），接下来 \(r\cdot d_\text{out}\) 个元素 reshape 为 \(B\)。
- 输出 \(W'\in\mathbb R^{d_\text{out}\times d_\text{in}}\)，float32。

## 提示

- 先生成固定排列，再切片、reshape、做低秩乘，最后加回 \(W_0\)。
- \(\alpha=0\) 或对应 \(\theta\) 位置全零时，\(W'\) 应严格等于 \(W_0\)。
- 同一 `proj_seed` 下改变 `layer_id` 会得到不同的 \(A/B\) 切片，这是为了模拟全模型共享参数时的层间不重叠分配。
