# PEFT：RandLora 冻结随机低秩基与可训系数——算法详解

## 1. 公式与出处

RandLora 出自 arXiv:2502.00987，核心思想是用**冻结的随机低秩基**替代可学习的 \(A/B\)，只学习一组组合系数 \(c\in\mathbb R^r\)：

\[\Delta W = \frac{\alpha}{r}\,B\,\operatorname{diag}(c)\,A.\]

其中 \(B\in\mathbb R^{d_\text{out}\times r}\)、\(A\in\mathbb R^{r\times d_\text{in}}\) 是固定种子生成的标准正态随机矩阵，训练时不再更新。每层可训练参数量从标准 LoRA 的 \(r(d_\text{in}+d_\text{out})\) 降到 \(r\)。

## 2. 随机基的确定性生成

- 统一使用 `torch.Generator().manual_seed(base_seed)`。
- 生成顺序必须固定：先按 `(d_out, r)` 生成 \(B\)，再按 `(r, d_in)` 生成 \(A\)。
- 由于生成器状态按顺序推进，任何顺序改变都会导致 \(A/B\) 与期望不一致。

## 3. 退化与判题考点

- \(c=0\)：\(\Delta W=0\)，\(W'=W_0\)。
- \(\alpha=0\)：\(\Delta W=0\)，\(W'=W_0\)。
- 不同 `base_seed` 产生不同随机基，从而 \(W'\) 不同。
- 输出只依赖 \(W_0, c, \alpha, r, \text{base_seed}\)，无运行时随机性。

## 4. 与现有题的分工

- `peft-lora-forward`/`peft-lora-merge`：\(A/B\) 均可训练。
- 本题：\(A/B\) 冻结随机、只训系数 \(c\)，强调可训练参数量与随机基重建。
