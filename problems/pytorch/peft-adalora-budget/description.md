# PEFT：AdaLoRA 预算内秩重分配

## 题目描述

**AdaLoRA**（Zhang et al., ICLR 2023, *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*；PEFT 库 `tuners/adalora`）把 LoRA 的加性更新重参数化为 SVD 形式的三元组：每层维护右奇异向量 \(A\in\mathbb R^{r\times d_\text{in}}\)、奇异值 \(E\in\mathbb R^{r\times 1}\)、左奇异向量 \(B\in\mathbb R^{d_\text{out}\times r}\)。它真正的新意在**训练中动态分配秩预算**：每隔若干步用「参数敏感度」给 \(r\) 个奇异三元组打分，把分数最低的若干组的奇异值置零——被置零的三元组不再贡献更新，等效于把预算从"不重要"的奇异方向挪给"重要"的方向。

敏感度评分（PEFT `RankAllocator` 口径）对每个标量参数 \(p\) 用两步梯度 \(g_1,g_2\) 做 EMA 平滑，\(\text{ipt}=|p\cdot g|\)：

\[\bar m^{(1)}=(1-\beta_1)\,\text{ipt}_1,\qquad u^{(1)}=(1-\beta_2)\,\bigl|\text{ipt}_1-\bar m^{(1)}\bigr|\]
\[\bar m^{(2)}=\beta_1\bar m^{(1)}+(1-\beta_1)\,\text{ipt}_2,\qquad u^{(2)}=\beta_2 u^{(1)}+(1-\beta_2)\,\bigl|\text{ipt}_2-\bar m^{(2)}\bigr|\]
\[\text{element}(p)=\bar m^{(2)}\cdot u^{(2)}\]

两步都从零状态起算（第一步没有历史项）。三个张量的 element 分数再按**维度平均**聚合成每个奇异三元组的总分：

\[s^E_i=\text{element}(E)_i,\qquad s^A_i=\frac{1}{d_\text{in}}\sum_{j}\text{element}(A)_{ij},\qquad s^B_i=\frac{1}{d_\text{out}}\sum_{w}\text{element}(B)_{wi}\]

\[\text{score}_i=s^E_i+s^A_i+s^B_i\]

预算截断取 \(r\) 个分数中**全局第 \(k\) 小**作阈值（\(k=r-\text{budget}\)，跨三元组全局比较），所有 \(\text{score}_i\le\) 阈值的三元组都被置零 \(E_i\gets 0\)——并列会多删，这是 PEFT 原生行为；\(k<1\)（budget \(\ge r\)）时不做任何掩码。注意掩码只落在 \(E\) 上：\(A,B\) 保持原样，被删方向只是"权重归零"。

## 任务

实现 AdaLoRA 的一次预算重分配：给定三元组当前参数与两步梯度、EMA 系数和预算，返回每个三元组的敏感度分数与掩码后的 \(E\)。

- 输入 \(E\in\mathbb R^{r\times 1}\)，\(A\in\mathbb R^{r\times d_\text{in}}\)，\(B\in\mathbb R^{d_\text{out}\times r}\)；梯度 \(g^{(1)}_E,g^{(2)}_E\) 同 \(E\) 形、\(g^{(1)}_A,g^{(2)}_A\) 同 \(A\) 形、\(g^{(1)}_B,g^{(2)}_B\) 同 \(B\) 形；\(\beta_1,\beta_2\in(0,1)\)；\(\text{budget}\) 为整数（\(0\le\text{budget}\le r\)）。
- 输出 \((\text{scores},\ \text{masked\_E})\)：\(\text{scores}\in\mathbb R^{r}\)，\(\text{masked\_E}\in\mathbb R^{r\times 1}\)（均 float32）。

**约定**：内部用 float64 计算、输出 float32；阈值判定用 \(\le\)（含并列）；不使用任何现成 PEFT 库。容差 \(10^{-5}\)。

## 提示

- 两步 EMA 从零状态开始：第一步的 \(\bar m,u\) 没有历史项，不要把第二步的递推套到第一步。
- \(s^A\) 对 \(A\) 的**行**（每个三元组的 \(d_\text{in}\) 个参数）取平均，\(s^B\) 对 \(B\) 的**列**取平均，方向别搞反。
- 全局第 \(k\) 小可用 `torch.kthvalue(scores, k)`，其中 \(k=r-\text{budget}\)。
- 并列分数会让删除数超过 \(r-\text{budget}\)——保留 PEFT 原语义，不要去重。
