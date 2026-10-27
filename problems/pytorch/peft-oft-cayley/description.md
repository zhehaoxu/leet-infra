# PEFT：OFT 正交微调的 Cayley 参数化

## 题目描述

**OFT**（Orthogonal Finetuning，Qiu et al., arXiv:2306.03889；PEFT 库 `tuners/oft`）的动机：文生图模型里神经元的**方向**编码了语义结构，微调应当**保范数、转方向**——于是它不学一个加性增量，而是学一个**正交矩阵**去旋转权重：

\[W' = R\,W_0,\qquad R^\top R = I\]

为了把"任意矩阵"约束成"正交矩阵"，OFT 用 **Cayley 参数化**：每块先由一组上三角参数 \(v\) 构造**反对称矩阵** \(Q\)（\(Q^\top=-Q\)），再做 Cayley 变换

\[R_b = (I+Q)\,(I-Q)^{-1}\]

反对称阵的 Cayley 变换恒为正交阵（证明见详解），且 \(v=0\Rightarrow Q=0\Rightarrow R_b=I\)——零初始化时适配器是恒等旋转，与 LoRA 系的"零初始化不扰动基座"异曲同工。

真实层的维度往往太大，OFT 把输出维切成若干**块**，每块独立旋转：设块大小 \(h\)、块数 \(n\)，把各块的 \(R_b\) 沿对角拼成块对角正交阵

\[R = \operatorname{blockdiag}(R_1,\dots,R_n)\in\mathbb R^{(n h)\times(n h)}\]

再乘到权重上。PEFT 实现细节：每块的参数是该块上三角（不含对角）的 \(h(h-1)/2\) 个标量，按 `torch.triu_indices(h, h, 1)` 的行主序排列；`Q` 由"上三角填 \(v\) 再减自身转置"得到；PEFT 默认用 Neumann 级数 \(I+2Q+2Q^2+\cdots\) 近似 Cayley 变换（同一极限，见详解），本题直接用闭式。

## 任务

给定块参数 `blocks`（形状 `(n_blocks, n_elem)`，`n_elem = h(h-1)/2`，`h` 由它唯一反推）与冻结基座 `W0`（形状 `(n_blocks*h, d_in)`）：

1. 每块：上三角填 \(v\) → 减转置得反对称 \(Q\) → Cayley 变换 \(R_b=(I+Q)(I-Q)^{-1}\)；
2. 拼块对角阵 \(R\)，输出 **`(R, W')`**，其中 \(W'=R@W_0\)。

`R` 与 `W'` 均为 float32 张量；结果按容差 \(10^{-5}\) 判定。

## 提示

- \(h\) 的反推：\(h = \dfrac{1+\sqrt{1+8\,\texttt{n\_elem}}}{2}\)。
- 求逆可用 `torch.linalg.inv` 或解线性方程组，二者数值上一致。
- 自检：`R @ R.T ≈ I`，且正交变换保持 Frobenius 范数 \(\|W'\|_F=\|W_0\|_F\)。
