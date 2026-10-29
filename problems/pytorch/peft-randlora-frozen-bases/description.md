# PEFT：RandLora 冻结随机低秩基与可训系数

## 题目描述

**RandLora**（arXiv:2502.00987）把 LoRA 的 \(A/B\) 因子设为**冻结的随机矩阵**（训练过程中不再更新），只学习一个长度等于秩 \(r\) 的**系数向量** \(c\in\mathbb R^r\) 来组合这些基：

\[\Delta W = \frac{\alpha}{r}\,B\,\operatorname{diag}(c)\,A,\]

其中 \(A\in\mathbb R^{r\times d_\text{in}}\)、\(B\in\mathbb R^{d_\text{out}\times r}\) 由 `base_seed` 通过 `torch.randn` 确定性地生成，\(c\) 是输入的可训练系数。由于 \(A/B\) 冻结，每层真正的可训练参数量只有 \(r\) 个标量。

## 任务

给定冻结基座 \(W_0\in\mathbb R^{d_\text{out}\times d_\text{in}}\)、系数向量 \(c\in\mathbb R^r\)、缩放 \(\alpha\)、秩 \(r\) 与随机种子 `base_seed`，按以下步骤返回合并权重 \(W'\)：

1. 用 `torch.Generator().manual_seed(base_seed)` 生成 \(B\sim\mathcal N(0,1)\)（形状 \((d_\text{out},r)\)），再生成 \(A\sim\mathcal N(0,1)\)（形状 \((r,d_\text{in})\)）。注意**先 B 后 A** 的顺序。
2. 计算 \(\Delta W = (\alpha/r)\,B\,\operatorname{diag}(c)\,A\)。
3. 返回 \(W' = W_0 + \Delta W\)。

- \(c\) 为零向量时 \(\Delta W=0\)，\(W'=W_0\)。
- \(\alpha=0\) 时同样退化为 \(W_0\)。
- 输出 \(W'\in\mathbb R^{d_\text{out}\times d_\text{in}}\)，float32。

## 提示

- 随机基的生成顺序必须固定：先按 `(d_out, r)` 生成 \(B\)，再按 `(r, d_in)` 生成 \(A\)；否则种子相同也会得到不同矩阵。
- 可用 `B @ torch.diag(c) @ A` 实现按系数缩放各秩-1 分量；当 \(r\) 较大时等价于 `B * c[None, :] @ A`。
- 本题只需重建并合并，不训练任何参数。
