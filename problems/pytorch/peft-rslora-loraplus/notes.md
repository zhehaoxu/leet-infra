# PEFT：rsLoRA 缩放与 LoRA+ 差异化学习率——算法详解

## 1. rsLoRA：缩放为什么是 α/√r

Kalajdzievski（arXiv:2312.03732）把 LoRA 前向写成 \(h=W_0x+\frac{\alpha}{r}BAx\) 并追问：秩 \(r\) 增大时更新项的尺度如何变化？结论是 \(\alpha/r\) 让更新有效范数以 \(1/r\) 衰减，比权重矩阵初始化固有的 \(1/\sqrt d\) 尺度衰减得更快，于是高秩时 LoRA 分支"越来越没用"、训练反而不稳。改为 \(\alpha/\sqrt r\) 后与标准参数化的尺度定律一致。论文实验：高秩（\(r\ge 16\)）下 rsLoRA 的 loss 明显更低、训练更稳；低秩时两种缩放几乎无差。注意 rsLoRA **不改变参数、梯度或优化器**，只是前向常数——与 merge 完全兼容：\(\Delta W=\frac{\alpha}{\sqrt r}BA\)。

## 2. LoRA+：B 侧学习率 ×16

Hayou et al.（arXiv:2402.12354）用有限宽度理论分析 LoRA 训练动力学：\(A\) 把输入投到低秩空间（下投影）、\(B\) 再投回输出空间（上投影），两者的梯度尺度天然不对称；共用 \(\eta\) 时部分层的特征学习率退化、\(B\) 学不动。修正是 \(\eta_B=\lambda\eta_A\)，推荐 \(\lambda=16\)（消融显示 16 在多数任务最优、且让 \(\eta_A\) 的选择更鲁棒）。LoRA+ 与缩放选择（\(\alpha/r\) 或 \(\alpha/\sqrt r\)）**正交**，可叠加使用。

## 3. 本题的合成口径（判题考点）

一步 SGD：\(A'=A-\eta_A g_A\)，\(B'=B-\lambda\eta_A g_B\)；输出更新后因子的乘积 \(B'A'\) 的两种缩放：

\[\Delta_\text{rs}=\frac{\alpha}{\sqrt r}B'A',\qquad \Delta_\text{vanilla}=\frac{\alpha}{r}B'A'\]

- `check_ratio_sqrt_r`：同一 \(B'A'\) 的两种缩放，任意非零元素处比值恒为 \(\sqrt r\)（\(r=9\) → 比值 3）。
- `check_zero_grad_no_change`：\(g_A=g_B=0\) 时没有更新，两输出应等于缩放后的**原始** \(BA\)——考"输出 = 更新后因子乘积"这个语义，而不是"输出 = 本步梯度贡献"。
- `check_a_side_unaffected_by_lam`：\(g_B=0\) 时换 \(\lambda\)（1 → 16）输出逐位不变——\(\lambda\) 只进 \(B'\) 的更新式，这是 LoRA+ 不对称性的最小断言。
- `check_example` 里 \(g_A,g_B\) 各只有一个非零元素，但 \(\lambda\) 的效果会通过 \(B'\) 与 \(A'\) 的**乘积耦合**扩散到 \(\Delta\) 的所有列——单看 \(B\) 的变化列是学不会的。

## 4. 与本合集其他题的分工

- `peft-lora-forward`/`peft-lora-merge`：静态 \(\alpha/r\) 缩放的前向与合并。
- `peft-adalora-budget`：改的是秩本身（预算重分配）；本题秩固定，改的是尺度（rsLoRA）与学习率（LoRA+）。
- 三者在真实训练里可以同时启用：AdaLoRA 管秩、rsLoRA 管缩放、LoRA+ 管优化器。

## 5. 验证方法

`check_example` 手算（\(r=4\)、\(\eta_A=0.1\)、\(\lambda=16\)、\(\alpha=8\)）：\(A\) 仅 \(A_{00}\) 有梯度、\(B\) 仅 \(B_{01}\) 有梯度 → \(A'_{00}=0.9\)、\(B'_{01}=-0.6\) → \(B'A'=[[3.9,0.4]]\) → \(\Delta_\text{rs}=[[15.6,1.6]]\)、\(\Delta_\text{vanilla}=[[7.8,0.8]]\)；`check_matches_oracle` 与纯 python 逐标量重算对拍（float64 内部、float32 输出、容差 \(10^{-5}\)）。
