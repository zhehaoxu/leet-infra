# PEFT：rsLoRA 缩放与 LoRA+ 差异化学习率

## 题目描述

LoRA 的更新量 \(\Delta W=\frac{\alpha}{r}BA\) 里藏着两个被后续工作各自修正的"超参陷阱"：

- **rsLoRA**（Kalajdzievski, arXiv:2312.03732）指出 \(\alpha/r\) 在**高秩**下会让更新有效范数以 \(1/r\) 衰减，比权重矩阵固有的 \(1/\sqrt d\) 尺度掉得更快——LoRA 的稳定性随秩增大而失衡。把缩放改成 \(\alpha/\sqrt r\) 后，高秩训练更稳、收敛更快。
- **LoRA+**（Hayou et al., arXiv:2402.12354）指出 \(A\)（下投影）与 \(B\)（上投影）共用一个学习率时，有限宽度网络的**特征学习率**不对称：\(B\) 侧需要更大的学习率 \(\eta_B=\lambda\,\eta_A\)（论文推荐 \(\lambda=16\)），否则低秩微调欠拟合。

两者机制不同（推理期缩放 vs 训练期优化器），但都能落到"一步 SGD 后的更新量"上观察：

\[A'=A-\eta_A\,g_A,\qquad B'=B-\lambda\,\eta_A\,g_B,\qquad \Delta W=B'A'\]

## 任务

给定 LoRA 因子与一步梯度，实现 SGD 更新，并分别输出 rsLoRA 缩放与普通缩放下的更新量。

- 输入 \(A\in\mathbb R^{r\times d_\text{in}}\)，\(B\in\mathbb R^{d_\text{out}\times r}\)，梯度 \(g_A,g_B\) 同形；\(\eta_A=\texttt{lr}\)、\(\lambda=\texttt{lam}\)、\(\alpha=\texttt{alpha}\) 为实数，\(r\) 为正整数。
- 输出 \((\Delta_\text{rs},\ \Delta_\text{vanilla})\)，均为 \(\mathbb R^{d_\text{out}\times d_\text{in}}\)：

\[\Delta_\text{rs}=\frac{\alpha}{\sqrt r}\,B'A',\qquad \Delta_\text{vanilla}=\frac{\alpha}{r}\,B'A'\]

**约定**：内部用 float64 计算、输出 float32；\(\sqrt r\) 用 float64 平方根；不使用任何现成 PEFT 库。容差 \(10^{-5}\)。

## 提示

- \(A\) 侧用 \(\eta_A\)，\(B\) 侧用 \(\lambda\eta_A\)——\(\lambda\) 只进 \(B\) 的更新式。
- 两个输出共享同一个 \(B'A'\)，只差标量因子 \(\sqrt r\) 倍。
- \(g_B=0\) 时换任何 \(\lambda\) 都不应改变输出（LoRA+ 的不对称性）。
