# PEFT：AdaLoRA 预算内秩重分配——算法详解

## 1. 公式与出处

AdaLoRA（Zhang et al., ICLR 2023, arXiv:2303.10512）观察到一个事实：把同一个参数预算 \(\text{budget}=r\cdot(d_\text{out}+d_\text{in}+1)\) 均匀地分给所有层并不是最优的——不同层、不同奇异方向的重要性差异很大。它把每层的增量更新写成 SVD 三元组：

\[\Delta W = \bigl(B\,(A\odot E)\bigr)\cdot\frac{s}{\rho+\epsilon},\qquad A\in\mathbb R^{r\times d_\text{in}},\ E\in\mathbb R^{r\times 1},\ B\in\mathbb R^{d_\text{out}\times r}\]

其中 \(A,B\) 是奇异向量（正交化近似），\(E\) 是奇异值，\(\rho\) 是当前秩数、\(s\) 是缩放（PEFT `get_delta_weight` 原式：`transpose(lora_B @ (lora_A * lora_E), fan_in_fan_out) * scaling / (ranknum + 1e-5)`）。三元组参数量 \(r(d_\text{out}+d_\text{in}+1)\) 比普通 LoRA 多 \(r\)，但换来**训练中重分配秩**的能力。

本题只考 `RankAllocator.update_and_allocate` 的评分与截断两步，不考前向。

## 2. 敏感度评分的两步 EMA（源码口径）

PEFT `RankAllocator` 对每个参数维护两个平滑量（对照源码变量 `exp_avg_ipt`/`exp_avg_unc`）：

- **重要性**：\(\text{ipt}=|p\cdot g|\)（参数与梯度的逐元素乘再取绝对值，即"梯度模长加权"）。
- **不确定度**：用 \(\text{ipt}\) 与其 EMA 均值 \(\bar m\) 的偏差 \(|\text{ipt}-\bar m|\) 再做一层 EMA。

两次 `update_and_allocate` 之间恰好积累两步新梯度，于是评分时从零状态连推两步：

\[\bar m^{(1)}=(1-\beta_1)\,\text{ipt}_1,\qquad u^{(1)}=(1-\beta_2)\,|\text{ipt}_1-\bar m^{(1)}|\]
\[\bar m^{(2)}=\beta_1\bar m^{(1)}+(1-\beta_1)\,\text{ipt}_2,\qquad u^{(2)}=\beta_2 u^{(1)}+(1-\beta_2)\,|\text{ipt}_2-\bar m^{(2)}|\]

注意 \(\bar m^{(1)}=(1-\beta_1)\text{ipt}_1\ne\text{ipt}_1\)（除非 \(\beta_1=0\)），所以 \(u^{(1)}=(1-\beta_2)\,\beta_1\,\text{ipt}_1\) 一般非零——"第一步没有历史"不等于"第一步没有不确定度"。最后 element 分数 \(=\bar m^{(2)}\cdot u^{(2)}\)：均值大、波动也大的参数才算"重要"，只大不波动的参数会被平权掉。

## 3. 三元组合成与全局截断（判题考点）

一个奇异三元组 \(i\) 的三个成员张量各有各的 element 分数，合成时**按各自的主维度平均**：

- \(E\) 本来就是每行一个标量：\(s^E_i=\text{element}(E)_i\)；
- \(A\) 的第 \(i\) 行（\(d_\text{in}\) 个参数）属于三元组 \(i\)：\(s^A_i=\text{mean}_j\,\text{element}(A)_{ij}\)（源码 `mean(dim=1)`）；
- \(B\) 的第 \(i\) **列**（\(d_\text{out}\) 个参数）属于三元组 \(i\)：\(s^B_i=\text{mean}_w\,\text{element}(B)_{wi}\)（源码 `mean(dim=0)`）。

\(\text{score}_i=s^E_i+s^A_i+s^B_i\) 是三者**直接相加**（不是平均）。截断在全局做：\(k=r-\text{budget}\)，阈值 \(=\) 第 \(k\) 小的分数（`torch.kthvalue(scores, k)`），掩码条件是 `scores <= threshold`——这意味着：

- **并列会多删**：两个三元组分数严格相等且恰好压在阈值上时，都会被置零，实际保留数会低于 budget。`check_ties_delete_more` 验证的就是这个边界（这也是 PEFT 原生行为，不去重）。
- **掩码只作用于 \(E\)**：\(A,B\) 原样保留。被删方向在后续训练里仍可能因为梯度流经 \(B\,(A\odot 0)\) 之外的路径而"复活"——论文的消融显示这种软删除比硬删参数更稳。
- \(k<1\)（budget \(\ge r\)）时 `kthvalue` 无定义，PEFT 直接跳过掩码，`check_budget_boundary` 的 `budget=r` 分支验它。

## 4. 与本合集其他题的分工

- `peft-lora-forward`/`peft-lora-merge`：静态 LoRA 的前向与合并，秩固定。
- `peft-lora-memory`：同样做"敏感度打分"，但那题是**层间**的静态参数计数口径（每层 score 只决定是否整层保留），本题是**层内**的动态秩重分配（每步都可能在同一层里删掉部分奇异方向）。
- 本题：AdaLoRA 的预算调度（cubic schedule 从 init 预算衰减到 target）被简化为单次给定 budget；两步梯度已直接给出，不用真的反向传播。

## 5. 验证方法

- `check_example`：\(r=2,d_\text{in}=d_\text{out}=1\)、两步梯度全 1、\(\beta=0.5\)，手算得 scores \([6.5625, 10.5]\)，budget=1 时 \(E_0\) 被置零。
- `check_matches_oracle`：随机输入与逐标量独立重算对拍（float64 内部、float32 输出、容差 \(10^{-5}\)）。
- 阈值与并列行为用 `check_ties_delete_more` 固定反例断言。
