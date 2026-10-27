# KV Cache：FastGen 头型分类与差异化保留预算

## 题目描述

FastGen（Ge et al., ICLR 2024, arXiv:2310.01801）的核心洞察是：**不同注意力头对缓存的需求结构完全不同**，用同一套 top-k 规则一刀切既浪费又伤质量。论文把头的注意力画像分成几类，并给每类配不同的压缩策略：

- **specialization 头**：注意力集中在少数固定 token（分隔符、标点、模板槽位）上，画像是尖峰——应保留**分数最高**的那几个；
- **retrieval 头**：注意力广泛铺在整段 prompt 上（需要回溯原文），画像平坦——应**多保留**，压缩率最小；
- **streaming 头**：注意力集中在最近的窗口（局部语法），画像右偏——只需 **sink + 最近 window**，位置规则即可，不用打分。

本题给出一个**完全确定性的玩具版判定**（真实 FastGen 用动态规划在候选策略集上搜最优，那是训练/校准期的一次性开销，不适合单题判题）。给定每个头的累计注意力行 \(s_h\in\mathbb R^{L}\)（非负），先算两个画像统计量：

\[
R_h=\frac{\sum_{j\ge L-w}s_h[j]}{\sum_j s_h[j]},\qquad
C_h=\frac{\max_j s_h[j]}{\sum_j s_h[j]}
\]

其中 \(R_h\) 是**近因质量占比**（最后 \(w\) 列占全部质量的比例），\(C_h\) 是**集中度**（最大单列占比）。分类按固定优先级：

\[
\text{type}_h=
\begin{cases}
\text{streaming (0)}, & R_h\ge r_{\text{hi}}\\
\text{specialized (1)}, & R_h<r_{\text{hi}}\ \wedge\ C_h\ge c_{\text{hi}}\\
\text{retrieval (2)}, & \text{otherwise}
\end{cases}
\]

三种类型的保留集（\(n_s\) 为 sink 数）：

| 类型 | 保留集 | 预算 |
| --- | --- | --- |
| streaming (0) | 前 \(n_s\) 个 ∪ 最后 \(w\) 个 | 位置规则，无打分 |
| specialized (1) | 前 \(n_s\) 个 ∪ 其余 top-\(k_s\) | \(k_s\) 小 |
| retrieval (2) | 前 \(n_s\) 个 ∪ 其余 top-\(k_r\) | \(k_r\) 大 |

这就是「**按头分配不等预算**」：同一层里 streaming 头可能只留 5 个、retrieval 头留 20 个，而不是每头留 10 个。

## 任务

实现 `solve(scores, n_sink, window, k_spec, k_retr, recency_ratio, concentration)`：

- 输入：`scores`（float32 `(H, L)`，非负累计注意力）；`n_sink`；`window`（\(w\in[1,L]\)）；`k_spec`、`k_retr`（非负，允许大于候选数，此时全取）；`recency_ratio`（\(r_{\text{hi}}\)）、`concentration`（\(c_{\text{hi}}\)），均在 \([0,1]\)。
- 输出 `(head_type, keep, kept)`：`head_type` 为 `(H,)` int64（0/1/2）；`keep` 为 `(H, L)` bool；`kept` 为 `(H,)` int64（各头保留数）。
- top-k 的**并列取小下标**；sink 位不参与 top-k 候选（已被无条件保留）。
- 总分为 0 的头：\(R_h=C_h=0\)，归 retrieval。
- 参数越界（`window ∉ [1,L]`、`n_sink ∉ [0,L]`、`k < 0`、阈值 ∉ [0,1]、`scores` 含负值）抛 `ValueError`。

## 示例

```
L = 8，n_sink = 1，window = 2，k_spec = 2，k_retr = 4，r_hi = 0.5，c_hi = 0.25

h0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0, 3.0]   R = 5.0/5.6 = 0.89 → streaming
     keep = [T, F, F, F, F, F, T, T]                     （sink 0 + 末窗 6,7，共 3）
h1 = [3.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]   R = 0.09，C = 0.68 → specialized
     keep = [T, T, T, F, F, F, F, F]                     （sink 0 + top2 → 1,2 并列取小）
h2 = [1,1,1,1,1,1,1,1]                          R = 0.25，C = 0.125 → retrieval
     keep = [T, T, T, T, T, F, F, F]                     （sink 0 + top4 → 1..4）

head_type = [0, 1, 2]
kept      = [3, 3, 5]
```

## 约束

- 分类按 streaming → specialized → retrieval 的**顺序**判定：`R` 与 `C` 同时达标时归 streaming。
- streaming 头**完全忽略分数**：末窗内哪怕全为 0 也保留，中间段分数再高也不保留（这是「位置策略」与「打分策略」的本质差别）。
- `k_spec` 与 `k_retr` 是不同量级的预算：检索头保留更多是设计意图，不要用同一个 top-k 统一处理。
- top-k 只在实际存在候选时按 `min(k, L-n_sink)` 保留，不足不补。
