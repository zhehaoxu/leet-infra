# KV Cache：Sparse-dLLM 延迟一步的块内驱逐

## 题目描述

扩散 LLM（dLLM）一次并行解码一整个 block，而不是一个 token：block 内的 query 全部已知、彼此没有因果掩码，所以打分时必须用**双向**视角——块内所有 query 一起看缓存。同时，当前 block 的 KV 还停留在上一轮去噪的中间态、尚未定稿，不能拿它当候选。

Sparse-dLLM（[arXiv:2508.02558](https://arxiv.org/abs/2508.02558)）的 `CustomCache.filter_cache` 因此把驱逐**延迟一步**：先用当前 block 的 query 给缓存打分，但把当前 block 的 KV 整段剔出候选；打完分只保留 `keep_ratio` 比例的候选。与自回归驱逐不同，这里没有 sink、也没有最近窗口——首尾 token 一样可以被丢。

打分链路是：块内 query 取均值 → 与每个候选 key 点积 → 跨 head 平均 → 一维 max 池化平滑 → 取 top-k。

## 规则

实现

```
solve(k, q_block, bef_filtered_len, block_len, keep_ratio=0.5, pool_kernel_size=3)
```

入参：

- `k`：`(B, H_kv, L, D)` 的 float32/float64 缓存，含当前 block 的 KV；
- `q_block`：`(B, H_q, block_len, D)`，当前 block 的 query，`H_q` 必须是 `H_kv` 的整数倍（GQA）；
- `bef_filtered_len`：非负 int，当前 block 之前有多少个槽位，即 block 占 `[bef_filtered_len, bef_filtered_len + block_len)`；
- `block_len`：正整数，当前 block 长度，必须等于 `q_block.shape[2]`；且 `bef_filtered_len + block_len <= L`；
- `keep_ratio`：\((0, 1]\)，保留比例；
- `pool_kernel_size`：正奇数，池化窗口（上游固定 3）。

计算：

1. **延迟一步**：候选为 `cat(k[:, :, :bef], k[:, :, bef+block_len:], dim=2)`，长度为 \(S' = L - \text{block\_len}\)（必须 \(\ge 1\)）；当前 block 整段消失，前后两段按原顺序拼回。
2. **GQA**：把候选 KV head 按 `repeat_interleave(H_q // H_kv)` 展开，query head \(h\) 使用第 \(h // (H_q/H_{kv})\) 个 KV head（不是 tile）。
3. `avg_q = q_block.mean(dim=-2)`，`(B, H_q, D)`：块内全部 query 的算术均值，双向、无因果。
4. `scores = avg_q @ cand.transpose(-2, -1)`，`(B, H_q, S')`。
5. `heads_avg = scores.mean(dim=1)`，`(B, S')`：先跨 head 平均。
6. `pooled = max_pool1d(heads_avg, pool_kernel_size, stride=1, padding=pool_kernel_size // 2)`：边界外用 \(-\infty\) 填充，两端只与存在的邻居取 max；`kernel=1` 时退化为原分数。
7. `keep_num = int(S' * keep_ratio)`（向下取整，可以是 0）；`keep_idx = argsort(-pooled, stable=True)[:, :keep_num]`，同分取更小的候选下标。

返回 5 键字典：

- `avg_q`：`(B, H_q, D)`，输入 dtype；
- `scores`：`(B, H_q, S')`，逐 query head 的点积；
- `heads_avg`：`(B, S')`；
- `pooled`：`(B, S')`，池化后的最终分数；
- `keep_idx`：`(B, keep_num)` int64，被保留的**候选下标**（不是原始缓存下标）。

内部按 float64 计算，四个浮点键按输入 dtype 返回。以下情况抛 `ValueError`：`k`/`q_block` 不是四维、dtype 不是 float32/float64 或两者不同、batch 或 `head_dim` 不一致、`H_q` 不能被 `H_kv` 整除、整数参数不是 int、`block_len < 1`、`bef_filtered_len < 0`、`block_len` 与 `q_block` 的块长不符、`keep_ratio` 不在 \((0,1]\)、`pool_kernel_size` 不是正奇数、块窗口越界或块外没有候选。

## 示例

```
k        = (1, 1, 6, 2)                      # 位置 2、3 是当前 block（两个 9）
           [[[ 2,  0], [ 0,  1], [ 9,  9], [ 9,  9], [-1,  0], [ 0, -1]]]
q_block  = (1, 2, 2, 2)
           head0: [[1, 0], [1, 2]]           head1: [[0, 1], [0, 0]]
bef_filtered_len = 2, block_len = 2, keep_ratio = 0.5, pool_kernel_size = 3

cand     = 位置 [0, 1, 4, 5] = [[2,0], [0,1], [-1,0], [0,-1]]      # S' = 6 - 2 = 4
avg_q    = head0 [1, 1], head1 [0, 0.5]
scores   = head0 [2, 1, -1, -1]
           head1 [0, 0.5, 0, -0.5]
heads_avg = [1.0, 0.75, -0.5, -0.75]
pooled    = [1.0, 1.0, 0.75, -0.5]          # 两端单侧窗口
keep_num  = int(4 * 0.5) = 2
keep_idx  = [0, 1]                          # 对应原始位置 0、1
```

块里的两个 9 完全没有出现在 `scores` 里；`pooled[0] = max(1.0, 0.75)`、`pooled[3] = max(-0.5, -0.75)` 体现边界只与单侧邻居比较，全程没有补 0 把负分抬起来。

## 约束

- **延迟一步是整段剔除**：当前 `block_len` 个槽位全部不进候选，不是只去掉最后一行 query；剩余候选保持「block 前 + block 后」的原顺序。
- **双向块视图**：`avg_q` 是块内 query 的算术均值，块内 query 顺序怎么排结果都一样；没有 causal mask。
- **先跨 head 平均，再池化**：顺序反了（逐 head 池化再平均）在两头峰值位置不同的数据上结果不同。
- **池化边界是单侧窗口**：`max_pool1d` 用 \(-\infty\) 填充；若补 0，全负的一段会被整体抬到 0，分数全平。
- **keep 名额向下取整**：`keep_num = int(S' * keep_ratio)` 可以是 0，此时返回 `(B, 0)` 的空索引而不是报错。
- **没有 sink / 最近窗口豁免**：首尾 token 与中间 token 一视同仁，最强候选也可能落在第一个或最后一个位置。
- **GQA 用 `repeat_interleave`**：KV head \(h\) 服务连续的一段 query head；用 tile / repeat 会把组映射搞错。
- **`keep_idx` 是压紧后的候选下标**：还原到原始缓存位置需 `i if i < bef_filtered_len else i + block_len`。
- **同分取更小下标**：上游 `topk` 不定义同分顺序，本题钉死。
- 输入只读，内部 float64；不假设 `k` 里有 sink token，也不假设 block 在末尾。
