# PEFT：DARE 丢弃重缩放与 TIES 三步合并——算法详解

## 1. 任务向量与两篇论文

模型合并的公共语言是**任务向量** \(\delta=\theta_\text{ft}-\theta_0\)（TIES 论文，Yadav et al., arXiv:2306.01708）。本题输入直接给定 \(\delta_1,\delta_2\)，考察其上两类算子：

- **DARE**（Yu et al., arXiv:2311.03099）：随机丢 \(p\) 比例参数，存活值乘 \(1/(1-p)\) 保期望不变——理论上"同源模型可以丢掉绝大多数冗余增量再合并"；
- **TIES**：TRIM（按幅度保 top-k）→ ELECT SIGN（多数票定符号）→ DISJOINT MERGE（只平均与共识符号一致者）。

DARE-TIES 管线 = 先 DARE 再把残差喂给 TIES，是 mergekit 等工具链的常用组合。判题环境无第三方合并库，全部按题面口径纯张量实现。

## 2. 口径三坑（判题考点）

1. **DARE 的重缩放只作用于存活值**：\(\delta'=m\odot\delta/(1-p)\)。若错缩放成"均值层面补偿"或对全向量除 \((1-p)\)，被丢位置的 0 也会被算错期望——好在 0 乘任何数还是 0，真正会错的是把 \(1/(1-p)\) 写成 \(1-p\) 或丢掉 rescale。
2. **TIES 的 TRIM 是每向量独立**的：\(\delta_1'\) 保自己的 top-`keep_n`，\(\delta_2'\) 保自己的，各自展平、按 \((|\cdot|\) 降序、下标升序\()\) 取前 `keep_n`。并列裁决不确定会导致两份合法实现对不上。
3. **DISJOINT MERGE 的分母**：论文 Algorithm 1 是「与 elect 符号一致的任务数」\(|T_j|\)，**不是**任务总数。两任务时单侧幸存取原值（分母 1）、双侧同号取均值（分母 2）、冲突 \(\gamma=0\) 置 0。`check_single_survivor_disjoint_mean` 专测这一点。

## 3. 数值与确定性设计

- 真实 DARE 的 \(m\sim\mathrm{Bernoulli}(1-p)\) 是随机的；判题把掩码作**显式入参**（本合集确定性惯例），`p` 只通过 rescale 起作用。
- TRIM 用 `sorted(key=(-abs, index))`（Python 稳定排序）保证并列取小下标；oracle 用**选择排序逐个挑最大**，结构独立。
- `p=0`、掩码全 1、`keep_n=N` 的退化档把 TIES 收缩成「同号均值 / 冲突置零 / 单侧取原值」，可手算验证。
- 一正一负同幅的参数（\(\delta_1=+5,\delta_2=-5\)）elect 票数相消 \(\gamma=0\) → TIES 置 0，而 DARE 均值恰好也是 0——两算子在"完全干涉"处殊途同归，这是 TIES 论文动机（符号干涉是平均法掉点主因）的最小反例。

## 4. 与本合集其他题的分工

- `peft-lora-merge`：单 adapter 的 \(W_0+sBA\) 折叠，不涉及多任务向量。
- `#83⑧（多 adapter 批量热切换）`：batch 内多 adapter 的**前向**合成，运行时行为。
- 本题：**离线权重合并**（训练后、部署前），输出合并矩阵本身。
