# 参考做法：2C 对齐、前后块配对与因果负载的闭式核对

## 实现

1. `align = 2 * cp_degree`，`padded = ceil(seq_len / align) * align`，`chunk = padded // (2 * cp_degree)`。**必须按 `2C` 对齐**：zigzag 的前后配对要求块数是偶数。
2. 先切出 `2C` 个块，第 `c` 块是 `[c*chunk, (c+1)*chunk)` 里的原序列下标，越界写 `-1`。
3. 按布局拼局部序列：`contiguous` 取 `(2r, 2r+1)`，`zigzag` 取 `(r, 2C-1-r)`，都是前块在前。
4. `local_mask` 逐位置取 `idx >= 0`；`causal_pairs[r] = Σ (idx+1)` 只对非负下标求和。
5. `imbalance = max/min`，`min == 0` 时按「全零给 1.0、否则 inf」处理。

常见错误：

- **把 zigzag 写成「前半正序、后半倒序」**。真实规则是第 $r$ 块与第 $2C-1-r$ 块配对：以 $C=2$ 为例，rank 0 拿块 0、3，rank 1 拿块 1、2；倒序会让 rank 1 拿块 2、1，局部顺序和负载都对不上。
- **按 `C` 对齐而不是 `2C`**。`N=6, C=2` 按 `C` 对齐只补到 6，切不出 4 个等长块；上游的 metadata 构建器也明确用 `alignment = 2 * cp_size`。
- **把 padding 当 query 计数**。因果负载只对真实 token 求和；若按补齐后的全局位置逐槽累加，`N=6, C=2` 的 rank 0 会把尾部补齐位置 6、7 各算成 7、8 个 key，负载从 3 变成 18，恰好掩盖了失衡。
- **`min == 0` 直接做除法**。`C > N` 时必然有 rank 一个真实 token 都没有。
- **contiguous 写成「第 r 块」**。contiguous 每 rank 是两块 `2r`、`2r+1`，合起来才是一段长块；只取一块会让局部长度只有一半。

## 正确性检验

7 个 `check_*` 覆盖：手算示例（zigzag 18/18、contiguous 10/26）、90 组对拍（`N ∈ 1..13,15,16,17,31,32` × `C ∈ {1,2,3,4,8}` × 两种布局，与纯 Python oracle 逐项比较）、padding 账（增量、总数、有效下标集合恒为 `range(N)`）、zigzag 平衡性（无 padding 时每个 rank 的负载**严格相等**、`imbalance == 1.0`；contiguous 在有多个 rank 时失衡；`C=1` 两种布局完全一致）、块组成（每个 rank 的前后块边界与顺序）、空 rank（`C > N` 时 `pairs` 含 0、`imbalance` 为 `inf`）、非法输入。

oracle 走另一条路径：对每个原序列 token 反推 `(rank, slot, 局部位置)` 后填表，再单独遍历 token 累计负载；参考解则是先按块构造再按 rank 拼接。两者共享的是契约，不是代码结构。

变异测试：交换 zigzag 前后块顺序、zigzag 退化成 contiguous、padding 放到序列头部、mask 全 True、负载只数 token 个数、imbalance 用均值——六个 mutant 全部被抓到；其中「padding 放到头部」被 `padding_ledger`/`matches_oracle`/`empty_ranks` 三个用例同时抓住。

## 范围与来源

上游规则来自 NVIDIA Megatron-LM 的 context-parallel 布局代码：

- `megatron/core/context_parallel/utils.py::_get_batch_on_this_cp_rank_contiguous`：要求序列长度能被 `2 * cp_size` 整除，取
  `front = narrow(cp_rank * segment_len, segment_len)`、`back = narrow((2*cp_size - cp_rank - 1) * segment_len, segment_len)`，然后 `cat(front, back)`；对非 attention-mask 的序列张量则取连续块 `narrow(cp_rank * (seq / cp_size), seq / cp_size)`。
- `megatron/core/context_parallel/layout.py::_local_segment_ids`：zigzag 下 `front_start = cp_rank * segments_per_cp_half`、`back_start = (2 * cp_size - cp_rank - 1) * segments_per_cp_half`，即前块与对称的后块。
- `layout.py::_build_thd_zigzag_metadata`：`alignment = 2 * cp_size`，每段补到 `alignment` 的整数倍；`chunk_length = padded_len / (2 * cp_size)`，`destination_cp_rank = min(chunk_id, 2*cp_size - chunk_id - 1)`，`slot = chunk_id >= cp_size`，局部位置 = `seq_start // cp_size + slot * chunk_length + offset_in_chunk`。本题的单序列版本正是把这段逻辑落到一个序列上。
- 负载动机见 `zhuzilin/ring-flash-attention` 的 issue #2：4 卡时 `b0 | b1 | b2 | b3` 换成 `b0,b7 | b1,b6 | b2,b5 | b3,b4`，因果负载才对齐。

本题的简化与差异：只做单序列、不引入 `tp_size`（上游在 TP>1 时每 rank 的原子段数会变，且要求 TP 为偶数）；padding 统一放在序列尾部（上游对每个变长段各自补齐，见 `distributed-cp-varlen-zigzag-plan`）；不做任何通信量或注意力数值计算——环上的 softmax 已在 `distributed-ring-attention`，激活/通信闭合账已在 `distributed-sequence-parallel`，head 维切分（Ulysses）已在 `distributed-tp-split-rearrange`。
