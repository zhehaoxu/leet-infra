# 参考做法：逐段补齐、局部基址与跨段槽位映射

## 实现

1. 求每段长度 `lengths`，逐段补齐到 `2C`：`p_b = ceil(n_b / (2C)) * (2C)`；`P = Σ p_b`，`L = P // C`。
2. 前缀和得到 `cu_seqlens_padded`。
3. 逐段处理：`chunk = p_b // (2C)`，段间局部基址 `base = q_b // C`（`q_b` 是补齐后的段起点）。
4. 对段内第 `cid` 块：`rank = min(cid, 2C-1-cid)`，`slot = 0 if cid < C else 1`，写入位置
   `base + slot*chunk + k`，值是 `orig_start + cid*chunk + k`（段内偏移越界则 `-1`）。
5. `pad_total = P - N`；`pad_between_seqs = any(p_b != n_b)`。

常见错误：

- **先把整条流补到 `2C`，再统一切块**。上游是**逐段**补齐：每段都要能被切成 $2C$ 个等长块。`[0,3,5], C=2` 若只把总长 5 补到 8 再切，段边界 `[0,3]` 与 `[3,5]` 会被块边界切开，段内配对无从谈起。
- **段内块长用全流统一值**。块长是 `p_b/(2C)`，各段可以不同；用 `P/(2C)` 会把短段写越界。
- **局部基址写成段序号乘固定长度**。基址是补齐后前缀和除以 `C`（`q_b // C`），不是 `b * something`。
- **配对写成 `cid % C` 或 `cid` 取模**。必须是 `min(cid, 2C-1-cid)`；`cid = C` 时它等于 `C-1`，不是 `0`。
- **忘记把段尾补齐写成 `-1`**。把 `orig_start + offset` 无条件写进去会产生不存在的 token 下标。
- **`pad_between_seqs` 只看总 padding**。本题口径是「任一段被补齐」，与总 padding 非零等价，但语义上要跟随段边界变化；上游用 `cu_seqlens != cu_seqlens_padded` 判定。

## 正确性检验

6 个 `check_*` 覆盖：手算示例（`[0,3,5]` 跨段 padding 与 `[0,4]` 无 padding）、19 组对拍（不同段数/长度/`C`，与纯 Python oracle 逐项比较）、单段退化（与 zigzag 单序列布局一致）、padding 账（每段补齐量 `< 2C` 且是 `2C` 的倍数、总量、flag 与是否有段被补齐等价）、块归属（每个 rank 的前/后块分别来自第 `r`、`2C-1-r` 块）、非法输入。

oracle 用另一条路径：每段先生成补齐流，再用切片取第 `r`、`2C-1-r` 块、拼到 `base` 位置；参考解则逐块逐槽计算下标。两者只在契约上一致。

变异测试：对齐粒度改成 `C`、块长改成 `p_b/C`、基址不做除以 `C`、slot 恒为 0、padding 不置 `-1`、flag 恒 `False`、归属改成 `cid % C`——七个 mutant 全部被抓到；其中「flag 恒 False」只被示例/对拍/padding 账三个用例抓到，`single_sequence` 与 `rank_assignment` 与它无关。

## 范围与来源

规则逐行来自 NVIDIA Megatron-LM 的 `megatron/core/context_parallel/layout.py::_build_thd_zigzag_metadata`：

```text
alignment = 2 * cp_size
target_lengths = ceil(source_lengths / alignment) * alignment
chunk_lengths = padded_lengths / (2 * cp_size)
chunk_ids = offsets / chunk_lengths
destination_cp_ranks = min(chunk_ids, 2 * cp_size - chunk_ids - 1)
chunk_slots = (chunk_ids >= cp_size)
target_local_positions = target_sequence_starts / cp_size
                       + chunk_slots * chunk_lengths
                       + offsets_in_chunk
rank_order_positions = destination_cp_ranks * (target_token_count / cp_size)
                     + target_local_positions
source_indices = source_sequence_starts + offsets   (offsets < source_length, else -1)
rank_order_indices.scatter_(0, rank_order_positions, source_indices)
```

`megatron/core/context_parallel/utils.py::_get_batch_on_this_cp_rank_padded_zigzag` 再用 `rank_order_indices.view(cp_size, -1)[cp_rank]` 做 `index_select`，把 `-1` 位置填 0 / `padding_mask=True`。

本题的口径与上游差异：

- **只设 `tp_size = 1`**。上游在 TP>1 时还会给最后一段补 `tp_padding * cp_size` 个 token，使每 rank 的总量能被 TP 整除；本题不做这一步（题面不变量里 `p_b` 只被补到 `2C`）。
- `cu_seqlens` 用 Python list 而非 int32 张量；`rank_order` 用嵌套 list 而非一维扁平张量——本题按 `[C, L]` 直接给出。
- 只做计划，不做真正的 all-to-all：`contiguous_to_zigzag` / `zigzag_to_contiguous` 的数据搬运与 split sizes 不在本题范围；ring 上的注意力数值见 `distributed-ring-attention`，单序列负载见 `distributed-cp-zigzag-partition`。
