# 解题说明 · R-KV 冗余感知打分

## 一句话

驱逐分数 = **注意力分 − 冗余分**：注意力高但和别人重复的 token 该走。四步就是「窗口 softmax 均值 → max-pool」「两两余弦 → 代表位清零 → softmax」「0.07 加权融合」「topk + 窗口恒留」。

## 三项为什么这么算

**注意力项**：只取最后 `window_size` 条 query——它们是「当前最关心什么」的最新证据；对每条 query 在 key 维度 softmax 后按 query 平均，得到一个 0-1 之间的能量分布。`max_pool1d(k=7)` 只是把相邻 token 的分数抹平一点，避免单点抖动决定去留。

**冗余项**：`k_norm @ k_normᵀ` 的余弦矩阵，对角线清零（自己和自己不算冗余），`> 0.5` 记为「重复」。一组互相重复的 token 只需要留一个代表，其余的在分数上被扣掉：每行把 `retain_direction` 那一端的命中下标当代表位、把它的相似度清 0，再对**列**取均值——列均值正好是「有多少行认为我是冗余的」，softmax 之后就是这个 token 的冗余分。

**融合**：`mix_lambda = 0.07`。注意力项只值 0.07，冗余项占 0.93，所以 R-KV 的排序几乎完全由「你有多重复」决定，注意力只在同一冗余档位内当微调。这不是调参失误，是论文的取舍：**重复比低注意力更该被驱逐**。

## 判分点

- `attn` 的 key 切片是 `[:-window_size]`：窗口内的 key 不参与打分，但**参与**冗余矩阵与代表位计算；
- `redundancy` 是 `softmax` 之后才切窗口列，所以 `N` 项之和小于 1；
- 代表位是「命中下标行」的 max，非命中位补 0：命中为空时返回 0，`"first"` 的 min 几乎恒为 0；
- `final` 的符号：加注意力、减冗余，系数是 `λ` 与 `1-λ`；
- `kept` 前 `budget-window_size` 条按分数降序、后 `window_size` 条原序，同分取小下标。

## 与相邻题的区别

`kv-merging-cos-nearest` 也用余弦：它把余弦当**正**信号去找合并目标（越像越该合）；这里余弦是**负**打分项，越像越该被驱逐，而且不合并、只丢弃。`kv-h2o-heavy-hitters` 只有注意力项，本钱全在累积注意力上；本题多了一项冗余，颗粒是 token。

## 复杂度

注意力打分 \(O(H W N D)\)，余弦矩阵 \(O(H L^2 D)\)；`L = budget` 级别（默认 128）时后者可忽略，一旦回到未压缩的长 cache 就是二次项——上游因此每个 compress step 只做一次。

## 出处

`Zefan-Cai/R-KV`（arXiv 2505.24133）：

- `rkv/compression/r1_kv.py:37-67`：`_compute_scores`；L40-48 取最后 `window_size` 条 query 的 softmax 并 `mean(dim=-2)`；L51-56 `F.max_pool1d(kernel_size=7, padding=3, stride=1)`；L58-62 `cal_similarity(...)[:, :, :-window_size]`；L64-66 `attn*w - sim*(1-w)`；
- `rkv/compression/r1_kv.py:78-79`：`if kv_cache_len < self.budget: return`（本题把这个门写成 `ValueError`）；L86 `final_score.topk(self.budget - self.window_size)`；L172-183 按 topk 下标 gather 后再把最后 `window_size` 条接回去；
- `rkv/utils.py:6-30`：`compute_attention_scores`，L12-14 除以 `sqrt(head_dim)`，L23-26 是 GQA 的组内 max 池化（本题限定 `H_q == H_kv`，不走这条路）；
- `rkv/utils.py:33-66`：`cal_similarity`；L41 `key/(norm+1e-8)`；L43-44 对角线 `fill 0`；L46 `similarity_cos > threshold`；L55 `torch.max(indices, dim=-1)`；L65 `scatter_(-1, retain, 0)`；L66 `mean(-2).softmax(-1)`；
- 默认值在 `r1_kv.py:10-15`：`budget=128 / window_size=8 / kernel_size=7 / mix_lambda=0.07 / retain_ratio=0.1 / retain_direction="last"`，L20 `assert budget - window_size > 0`。

**本题自己的补充**：同分取小下标（上游 `topk` 未定义顺序）；`q` 只给最后 `window_size` 条并显式禁止 causal mask；`"first"` 分支的退化行为写进题面；`retain_ratio`（只服务 `*_percent` 分支）不在考纲内。

## 相关题目

- `kv-merging-cos-nearest`：余弦当正信号选合并目标，本题当负信号做驱逐；
- `kv-h2o-heavy-hitters`：只有注意力、按累积分数驱逐；
- `kv-snapkv-pooling`：观测窗内真实注意力 + 池化，没有冗余项。
