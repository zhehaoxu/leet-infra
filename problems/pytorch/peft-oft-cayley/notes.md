# PEFT：OFT 正交微调的 Cayley 参数化——算法详解

## 1. 公式与出处

OFT（Qiu et al., *Controlling Text-to-Image Diffusion by Orthogonal Finetuning*, arXiv:2306.03889；PEFT `tuners/oft/layer.py`）。与 LoRA 族「加性增量」不同，OFT 学**正交旋转**：\(W'=R W_0\)，\(R^\top R=I\)。它保持每个输出神经元的权重范数（频谱不变、能量不变），只转动方向——对扩散模型的语义结构特别友好。

## 2. Cayley 参数化：为什么反对称 → 正交

任意反对称阵 \(Q^\top=-Q\)，令 \(R=(I+Q)(I-Q)^{-1}\)。验证：

\[R^\top R=(I-Q)^{-\top}(I+Q)^\top(I+Q)(I-Q)^{-1}=(I-Q)^{-1}(I-Q)(I+Q)(I-Q)^{-1}=I\]

中间用到 \((I+Q)^\top=I-Q\) 与反对称阵和单位阵的**可交换性**（\(QI=IQ\) 显然）。几何上这是把「旋转角速度 \(\Omega\)」积分成「旋转 \(e^{\Omega t}\)」的离散化：Cayley 变换是矩阵指数 \(e^{Q}\) 的有理逼近（Padé-(1,1)），二者在 \(Q\to0\) 时同阶。

**零初始化**：\(v=0\Rightarrow Q=0\Rightarrow R=I\)，训练起点适配器是恒等旋转；LoRA 系置零因子、OFT 置零旋转，同一个"起点不扰动基座"哲学。

## 3. PEFT 实现口径

`OFTRotationModule` / `_pytorch_skew_symmetric`：

- 每块参数量 \(h(h-1)/2\)（严格上三角，不含对角——对角本来就必须为 0），按 `torch.triu_indices(h, h, 1)` 行主序展开；
- \(Q\)：先 `Q[rows, cols] = v`，再 `Q = Q - Q.T`（自动把下三角填成相反数、对角抵消为 0）；
- Cayley：PEFT 默认 `use_cayley_neumann=True`，用 **Neumann 级数** \(I+2Q+2Q^2+2Q^3+\cdots\)（`num_neumann_terms=5`）逼近。由于 \((I-Q)^{-1}=I+Q+Q^2+\cdots\)，闭式 \((I+Q)(I-Q)^{-1}=I+2Q+2Q^2+2Q^3+\cdots\)——级数与闭式是同一矩阵的截断/极限，本题直接用闭式，数值上与 PEFT 截断在 \(Q\) 小时一致到高阶项；
- PEFT 还有 `coft`（球面约束的 cos-theta 投影）与 `block_share`（各块共享同一 \(R_b\)）两个开关，本题不展开；
- 块对角：\(R=\operatorname{blockdiag}(R_1,\dots,R_n)\)，`check_block_diagonal_structure` 断言非对角块全零。

## 4. 判题红线与自检

- **只断言 \(R\) 与 \(W'\) 的数值**，不断言特征向量、旋转角符号——正交阵的参数化与特征分解在不同实现下符号/顺序可能不同，直接对拍必翻车（本合集通用红线）。
- 正交性自检：\(RR^\top=I\)、\(R^\top R=I\)；**Frobenius 范数守恒** \(\|RW_0\|_F=\|W_0\|_F\)（正交阵左右乘都不改变谱范数族）。
- 数值路径：主解 `inv`，oracle `solve` 解 \((I-Q)R=I+Q\)，两者在 float64 内一致到 \(10^{-12}\)，float32 输出容差 \(10^{-5}\) 绰绰有余。
- BOFT（2311.06243）把块对角推广成蝴蝶分解的连乘；PSOFT 只训 \(R-I\) 的低秩中心——同族方法在本题之外的批次处理。
