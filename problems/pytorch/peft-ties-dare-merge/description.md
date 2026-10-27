# PEFT：DARE 丢弃重缩放与 TIES 三步合并

## 题目描述

微调出多个同源模型后，把它们**合并成一个多任务模型**是推理侧的高频工程活。两个最常用的合并算子都以「任务向量」为输入（\(\delta=\theta_\text{ft}-\theta_0\)，本题直接给你 \(\delta\) 矩阵），务必严格按以下口径实现。

**第一步：DARE 丢弃重缩放**（Drop And REscale，Yu et al., arXiv:2311.03099《Language Models are Super Mario》）。任务向量里大量参数互相抵消、冗余度高，DARE 以保留概率 \(1-p\) 随机丢弃再放大存活值以保期望：

\[\delta' = \frac{m\odot\delta}{1-p},\qquad m\in\{0,1\}^{N}\ (m=1\ \text{表示保留})\]

本题**不抛硬币**：丢弃掩码 \(m_1,m_2\) 作为显式输入给出（保证确定性）。两任务 DARE 后取均值：

\[\tau_\text{DARE}=\tfrac12\bigl(\delta_1'+\delta_2'\bigr)\]

**第二步：TIES 合并**（TrIm, Elect Sign & disjoint merge，Yadav et al., arXiv:2306.01708），作用于 \(\delta_1',\delta_2'\)（即 DARE-TIES 完整管线），三步：

1. **TRIM**：每个向量展平后按 \(|\delta'|\) 保留最大的 `keep_n` 个，其余置零；**幅度并列时取下标小者**。
2. **ELECT SIGN**：逐位置投票 \(\gamma=\operatorname{sign}\bigl(\operatorname{sign}(\delta_1')+\operatorname{sign}(\delta_2')\bigr)\)；两向量一正一负（票数相消）或全零时 \(\gamma=0\)。
3. **DISJOINT MERGE**：逐位置只把**符号与 \(\gamma\) 一致**的向量纳入均值，**分母是"一致任务数"（1 或 2），不是任务总数**：

\[\tau_\text{TIES}[j]=\frac{\sum_{t:\,\operatorname{sign}(\delta_t'[j])=\gamma_j\neq0}\delta_t'[j]}{\bigl|\{t:\operatorname{sign}(\delta_t'[j])=\gamma_j\neq0\}\bigr|}\qquad(\gamma_j=0\Rightarrow 0)\]

这意味着"单侧幸存"的位置取幸存者**原值**而不是除以 2——这正是 TIES 区别于普通平均、能防符号干涉的关键。

## 任务

给定两个任务向量 \(\delta_1,\delta_2\)（同形矩阵）、丢弃概率 `dare_p`（\(0\le p<1\)）、两个 0/1 掩码 `mask1/mask2`（1=保留）、保留个数 `keep_n`（\(1\le\text{keep\_n}\le N\)，\(N\) 为元素总数），返回二元组 **`(dare_merged, ties_merged)`**：

- `dare_merged` = \(\tfrac12(\delta_1'+\delta_2')\)，其中 \(\delta_t'=m_t\odot\delta_t/(1-p)\)；
- `ties_merged` = 对 \(\delta_1',\delta_2'\) 施加 TIES 三步的结果。

均为 float32 张量，与输入同形；容差 \(10^{-5}\)。

## 提示

- `p=0` 且掩码全 1 且 `keep_n=N` 时：DARE 退化为普通平均；TIES 退化为"符号一致处平均、冲突处置零"。
- TRIM 的并列裁决是确定性关键：按 `(幅度降序, 下标升序)` 排序取前 `keep_n` 个。
