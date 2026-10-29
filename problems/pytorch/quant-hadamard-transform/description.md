# 量化：Hadamard 变换的快速递归构造

## 题目描述

**Hadamard 变换**（Walsh-Hadamard）在 QuaRot 等量化/旋转方法中用于把激活/权重旋转到各向同性的方向。规模为 \(2^k\) 的 Sylvester 矩阵递归定义为

\[H_1=[1],\qquad H_{2n}=\begin{bmatrix}H_n & H_n\\ H_n & -H_n\end{bmatrix}.\]

归一化后 \(H_n/\sqrt n\) 是正交矩阵：\((H_n/\sqrt n)^{\mathsf T}(H_n/\sqrt n)=I\)。

当维度 \(n\) 不是 2 的幂时，采用 **block-diagonal** 近似：把 \(n\) 贪心地拆成若干 2 的幂之和，每个块放一个对应规模的归一化 Hadamard 矩阵，块间用随机符号对角阵 \(D=\operatorname{diag}(s_1,\dots,s_n)\)（\(s_i\in\{-1,+1\}\)）做符号翻转。最终变换为

\[y=H^{(n)} D x.\]

随机符号由 `seed` 通过 `torch.Generator` 确定，保证可复现。

## 任务

实现 `solve(n, x, seed=0)`：

- `n`：正整数维度，与 `x.shape[-1]` 一致。
- `x`：float32，形状 \((n,)\) 或 \((b, n)\)。
- `seed`：非负整数，用于生成随机符号。
- 返回 `(H, y)`：
  - `H`：float32，形状 \((n, n)\)，是归一化后的 Hadamard / block-diagonal Hadamard 矩阵；
  - `y`：float32，形状与 `x` 相同，为 \(H D x\)。

## 提示

- 递归构造 \(H_{2n}\) 时可用 `torch.cat` 拼 `[H, H]` 与 `[H, -H]`，最后整体除以 \(\sqrt n\)。
- 非 2 的幂：从大到小拆分 \(n=2^{k_1}+2^{k_2}+\dots\)，用 `torch.block_diag` 拼接各子块。
- 随机符号用 `torch.randint(0, 2, (n,), generator=g) * 2 - 1` 生成。
