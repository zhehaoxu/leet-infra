# KV Cache：HOMER 层级 chunk 合并

## 题目描述

长文档 prefill 的显存随上下文线性增长。HOMER 把 context 切成 \(2^h\) 个 chunk 当叶子：每个叶子先跑若干层、按注意力剪到指定长度，兄弟结点再两两合并、再剪一次，直到根只剩一个 chunk。每一层只处理「数量减半、长度也减半」的 chunk，显存峰值只跟 chunk 长度有关，不跟文档长度有关。

单个结点做两件事：**合并**与**剪枝**。

合并（`Chunk.merge`）：两个 chunk 共享同一段 affix——开头 `prefix_len` 个 token、结尾 `suffix_len` 个 token 在每个 chunk 里都占同样的位置。合并时两个孩子的 prefix **逐元素取平均**、suffix 同样取平均，中间真正的 context 直接首尾拼接，顺序是 `[prefix 均值, 左 context, 右 context, suffix 均值]`，长度 \(M = M_L + M_R - \text{prefix} - \text{suffix}\)。

剪枝（`_merge_power_max_last_calibrated`）：取 `metric` 最后一行的注意力、按 head 取 max，再减去与位置相关的标定偏置

\[
\text{sig}(j) = \max_h \text{metric}[h, -1, j] - \text{bias}[\text{rel}(j)],\qquad \text{rel}(j) = (M-1) - j
\]

`rel` 是「离最后一个 token 有多远」，所以 `bias[0]` 打在最后一个位置上。减完偏置，把 `reduction_mask` 为 False 的位置（affix 保护区）置为 `+inf`，再按 `topk(largest=False)` 取最小的 `num_tokens_to_reduce` 个删掉——保护区是 `+inf`，永远进不了最小的那一批。

## 任务

实现 `solve(k_left, v_left, k_right, v_right, metric, bias, reduction_mask, num_tokens_to_reduce, prefix_len, suffix_len)`：

- `k_left`/`k_right`：`(M_L, H, D)` / `(M_R, H, D)`，NHD 布局；`v_*` 同形状同 dtype；
- `metric`：`(H, Q, M)`，**只有最后一行** `metric[:, -1, :]` 参与；
- `bias`：`(R,)`，`R >= M`，按「离末尾的距离」索引；
- `reduction_mask`：`(M,)` bool，True 表示可剪；
- `num_tokens_to_reduce`、`prefix_len`、`suffix_len`：非负整数，且 `prefix_len + suffix_len >= 1`。

内部一律 float64，返回按输入 dtype。

返回 4 键字典：

- `weights`：`(M,)`，减完偏置的显著性，保护区置 `+inf`（上游的返回值就带着这些 `inf`）；
- `removed`：`(r,)` int64 升序，被删掉的下标（同分取更小下标）；
- `merged_k` / `merged_v`：`(M-r, H, D)`，合并之后按原顺序留下。

`num_tokens_to_reduce = 0` 时一个都不删，长度仍是 `M`，权重照常计算。

非法输入抛 `ValueError`：非三维、左右 dtype 不同或不是 float32/float64、head 数或 head_dim 不一致、k/v 长度不一致、`metric` 不是 `(H, Q, M)`、`bias` 短于 `M`、`reduction_mask` 不是 `(M,)` 的 bool、参数不是非负整数、affix 全零或超过任一 chunk、`r >= M`、`r` 超过可剪位置数。

## 示例

```
k_left  = [[10], [12], [1], [2], [3], [40]]        # (6,1,1)：prefix 2 + context 3 + suffix 1
k_right = [[20], [30], [7], [8], [50]]             # (5,1,1)：prefix 2 + context 2 + suffix 1
prefix_len = 2, suffix_len = 1                     # 合并后 M = 6 + 5 - 3 = 8
v_left, v_right = k_left, k_right                  # 示例里让 K/V 一致
metric = [[[0.9, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]]     # (H,Q,M) = (1,1,8)
bias   = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
reduction_mask = [F, F, T, T, T, T, T, F]
num_tokens_to_reduce = 3

merged_k = [(10+20)/2, (12+30)/2, 1, 2, 3, 7, 8, (40+50)/2]
         = [15, 21, 1, 2, 3, 7, 8, 45]
rel      = [7, 6, 5, 4, 3, 2, 1, 0]
weights  = metric[0,-1,:] - bias[rel]
         = [0.55, 0.50, -0.15, 0.00, 0.15, 0.30, 0.45, 0.60]
         = [inf, inf, -0.15, 0.00, 0.15, 0.30, 0.45, inf]   # 保护区置 +inf
removed  = [2, 3, 4]                                        # 最小的三个
merged_k = [15, 21, 7, 8, 45]                               # 删完剩下的
```

权重最低的 −0.15、0.00、0.15 依次落在位置 2、3、4 上，正好被删掉；位置 0、1、7 的原始权重是 0.55、0.50、0.60，本来也排不进最小的三个，再加一层 `+inf` 保护，任何 `r` 都删不到它们。

## 约束

- **affix 取平均，不是拼两份**：两个孩子的 prefix 是同一段 token，拼起来会把 affix 复制 \(2^h\) 份；平均让它始终只占一份长度。
- **拼接顺序**是 `[prefix 均值, 左 context, 右 context, suffix 均值]`，不是「左 chunk 整体 + 右 chunk 整体」。
- **偏置从末尾数**：`bias[0]` 作用在最后一个位置，`bias[M-1]` 作用在第一个。
- **保护区是 `+inf` 而不是 0**：剪枝取的是最小值，0 照样可能最小；`+inf` 才挡得住。
- **同分取更小下标**：上游 `torch.topk` 不定义同分顺序，本题钉死。
- **`r = 0` 不剪**：直接返回合并后的 `M` 个 token，权重不变。
