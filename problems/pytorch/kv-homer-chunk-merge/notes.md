# 解题说明 · HOMER 层级 chunk 合并

## 一句话

把 context 切成 \(2^h\) 个 chunk 当叶子，兄弟结点「affix 取均值、context 拼接」，再按「最后一行注意力 − 位置偏置」删掉最小的 `r` 个 token。

## 合并为什么这么写

每个 chunk 都带着同一段 affix（`make_chunks` 里就是 `torch.cat([prefix_ids, context_ids[:, start:end], suffix_ids])`）。如果合并时直接把两个 chunk 首尾拼接，affix 会被复制 \(2^h\) 份，树的每一层长度都不降；取均值之后 affix 永远只占一份，长度才按 \(M_L + M_R - \text{affix}\) 收缩。上游 `Chunk.merge` 的顺序也钉死了这一点：`[prefix 均值, 左 context, 右 context, suffix 均值]`，hidden states（L218-232）与 KV cache（L268-299）用的是同一个顺序，reduction mask 也按四段拼（L331-341）。

## 剪枝为什么用 +inf

剪枝取的是**最小值**（`topk(largest=False)`）。保护区若只是「权重设 0」，当所有分数都为正时 0 反而最小，affix 立刻被删光；`masked_fill(~reduction_mask, math.inf)` 让它们永远排在末尾（L243-245）。注意返回值就是 masked 之后的张量（L252），所以 `weights` 在保护区上必须是 `+inf`，不是原始分数。

## 偏置为什么反向

上游先把 `position_ids` 减去最后一个位置再乘 −1，得到「离末尾的距离」（L229-234），然后 `bias_wrt_rel_pos[position_ids]`（L237-238）。所以 `bias[0]` 对应最后一个 token、`bias[M-1]` 对应第一个——顺着位置加会整体错位。

## 与上游的差别

- **只做一个结点**。整棵树是把这一步按高度递归：`_merge` 先对左右子树各求一次 `l_target_len`/`r_target_len`（L575-600，右侧过短时两者不等），再 `__process_nonleaf` 合并 + 剪枝。中间结点的 metric 来自它自己那几层 forward，本环境无法复现，因此 `metric` 是显式输入。
- 上游的 `r` 是 `hidden_states.shape[1] - target_len`（L405），且只在该结点最后一层设 reduction info（L412-417）；这里 `r` 直接作为入参。
- 上游合并后 `position_ids` 会跳变（`chunk.py` L393-398 把 suffix 偏到 `prefix + eff_chunk_len`），本题按**连续位置**取 `rel = (M-1) - j`。
- `r = 0` 时上游短路、把 `significance_weights` 填成占位全 1（L300-305）；本题照常算真实权重，只是不删。
- 同分取小下标、形状/参数校验是本题补充。

## 边界

- `r >= M` 报错：上游 `assert target_len > 0`，一个 chunk 至少要留一个 token。
- `r` 超过可剪位置数也报错：保护区是 `+inf`，`topk` 会先删到它们身上，这是不允许的。
- `prefix_len + suffix_len = 0` 与「affix 超过任一 chunk」都报错，对应上游 `assert prefix_len > 0 or suffix_len > 0`。

## 复杂度

合并是 \(O((M_L+M_R)HD)\) 的访存拷贝，剪枝是一次 \(O(M)\) 的 topk 加一次 gather。真正省下来的是 forward：每一层只对 \(2^{h}\) 个、每个长度减半的 chunk 跑 `layers_per_chunk` 层，总计算量是 \(O(h)\) 个 chunk 长度的和，而不是整篇文档。

## 出处

`alinlab/HOMER`（仓库没有 README、代码与注释里也没有 arXiv 编号，只引代码路径）：

- `src/homer/patch_llama.py` L222-252 `_merge_power_max_last_calibrated`：L227 `metric[:, :, -1, :].max(dim=1).values`、L229-234 rel 化、L237-238 取 bias、L240 相减、L243-245 `masked_fill(~mask, inf)`、L248-250 `topk(..., largest=False)`、L252 返回 `(indices, significance_weights)`；
- L256-307 `_apply_pruning`：L276 `if num_tokens_to_reduce > 0`、L290-292 `prune_mask[indices_to_remove] = False`、L300-305 `r == 0` 的占位分支；
- L389-453 `__forward_and_reduce`：L405 `r = hidden_states.shape[1] - target_len`、L412-417 只给最后一层设 reduction info、L442 `chunk.prune(prune_mask)`；
- `src/homer/chunk.py` L166-181 `prune`（用 `prune_mask` 索引 position/attention/reduction/cache）、L207-354 `merge`（L218-232 顺序、L268-299 KV 均值、L331-341 mask 拼接）、L427 `num_full_chunks = 2 ** ceil(log2(num_chunks))`；
- `patch_llama.py` L637-645 `tree_height`、`num_chunks = 2 ** tree_height`、`layers_per_chunk = floor(num_merging_layers / (tree_height + 1))`。

## 相关题目

- `kv-merging-cos-nearest`：按 key 的余弦相似度把被驱逐的 token 合并进最近邻，位置无关、单层；本题是**位置驱动的树形合并**，选谁删由最后一层注意力决定。
- `kv-quest-page-bound`：page 粒度的注意力上界；本题的显著性直接来自真实注意力的一行。
- `kv-qtrust-twohop`：同一批的 query 加权打分；HOMER 的 metric 只取最后一行，不做传播。
