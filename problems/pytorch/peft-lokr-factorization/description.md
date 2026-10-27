# PEFT：LoKr Kronecker 因子分解

## 题目描述

**LoKr**（Kronecker Product adaptation，Yeh et al., arXiv:2108.07383；PEFT 库 `tuners/lokr`）把权重增量表达成 **Kronecker 积** \( \Delta W = W_1 \otimes W_2 \)：要把 \(d_\text{out}\times d_\text{in}\) 的矩阵拆成两个因子 \(W_1\in\mathbb R^{a\times b}\)、\(W_2\in\mathbb R^{c\times d}\)，必须先回答——

\[a\cdot c = d_\text{out},\qquad b\cdot d = d_\text{in}\]

即把维度**因式分解**成两个乘子。PEFT 的 `factorization(dimension, factor)` 就是干这件事的，它的规则是本题的全部（务必严格按此实现）：

- **显式因子分支**：若 `factor > 0` 且 `factor` 整除 `dimension`，直接返回 `(factor, dimension // factor)`——**保持这个顺序，不做大小交换**（比如 `dimension=12, factor=6` 返回 `(6, 2)`，\(6>2\) 是合法的）。
- **自动分解分支**（`factor == -1`）：在所有满足 \(m\cdot n=\text{dimension}\) 且 \(m\le n\) 的因子对中，返回 **\(m+n\) 最小**（即 \(m\) 最接近 \(\sqrt{\text{dimension}}\)、分解最平衡）的那一对。例如 `256 → (16, 16)`、`128 → (8, 16)`、质数 `127 → (1, 127)`。
- **非法输入**：`dimension <= 0`、`factor` 不是 `-1` 也不是正数、或 `factor > 0` 但不整除 `dimension`，抛 `ValueError`。

分解越平衡（\(m,n\) 越接近），Kronecker 积两个因子的形状越方，LoKr 论文指出这通常对应更好的表达力-参数量折中；PEFT 默认 `factor=-1` 走的就是自动平衡分解。

## 任务

实现上述 `factorization` 规则：输入 `dimension`（正整数）与 `factor`（`-1` 或正整数），返回二元组 `(m, n)`（两个 Python 整数，\(m\cdot n=\text{dimension}\)）。非法输入抛 `ValueError`。

## 提示

- 自动分支可以 \(m\) 从 1 扫到 \(\sqrt{d}\)，维护「当前最小 \(m+n\)」的因子对；质数自然落在 \((1, d)\)。
- 显式分支**不要**画蛇添足地排序：`(6, 2)` 就是 `(6, 2)`。
