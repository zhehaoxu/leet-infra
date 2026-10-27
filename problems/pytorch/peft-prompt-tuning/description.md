# PEFT：Prompt-Tuning 软提示嵌入拼接

## 题目描述

**Prompt-tuning**（Lester et al., EMNLP 2021, arXiv:2104.08691）是 prefix-tuning 的极简版：只在**输入嵌入层**前面拼 \(L\) 个可学习的软提示 token，随嵌入一起进模型。T5-XXL 上它以每任务 \(L\cdot d\) 个参数追平全量微调（模型越大差距越小），且多任务共享同一个冻结模型——每任务只存一张小小的提示表。

前向：输入嵌入 \(E\in\mathbb R^{S_0\times d}\)，任务 \(t\) 的软提示取自共享池（多任务 serving 形态）：

\[X_t=\bigl[\text{pool}[\text{task\_ids}[t]];\ E\bigr]\in\mathbb R^{(L+S_0)\times d}\]

即先按行索引 gather 出该任务的 \(L\) 个提示行，再**前缀在前**拼接原嵌入。每任务的可训参数量 \(n_\text{params}=T\cdot L\cdot d\)（\(T\) 个任务）——与池大小 \(P\) 无关，池只是存储形态。

## 任务

给定软提示池、每任务的行索引与输入嵌入，产出拼接后的批量嵌入与参数量。

- 输入：\(\texttt{pool}\in\mathbb R^{P\times d}\)，\(\texttt{task\_ids}\in\mathbb Z^{T\times L}\)（每行 \(L\) 个互不相同的池行索引，\(0\le\text{idx}<P\)），\(E\in\mathbb R^{S_0\times d}\)。
- 输出 \((X,\ \texttt{n\_params})\)：\(X\in\mathbb R^{T\times(L+S_0)\times d}\)（float32，\(X[t,:L]=\text{pool}[\text{task\_ids}[t]]\)、\(X[t,L:]=E\)），\(\texttt{n\_params}=T\cdot L\cdot d\)。

**约定**：内部用 float64 计算、输出 float32；不使用任何现成 PEFT/transformers 库。容差 \(10^{-5}\)。

## 提示

- gather 是逐任务的行索引操作：`task_ids` 是二维的，第 \(t\) 行是任务 \(t\) 要取的 \(L\) 个池行。
- 拼接顺序：提示在前、原嵌入在后。
- \(n_\text{params}\) 只数"每任务独占视角"的可训参数 \(T\cdot L\cdot d\)，与 \(P\) 无关。
