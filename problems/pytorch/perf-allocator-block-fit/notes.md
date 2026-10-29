# 详解：缓存分配器 best-fit 块管理

## 出处
PyTorch `c10/cuda/CUDACachingAllocator.cpp` 的 `find_block`/`release_block`/`get_free_block`（2MB segment、small/large pool、512B 对齐）与 TensorFlow BFC allocator 是同一套思路的两种实现：**段（segment）+ 块表 + best-fit + 释放合并**。本题把段简化成一条连续地址空间、去掉 stream/事件与设备侧细节，只保留决定 reserved/碎片/扩段次数的整数逻辑——这三个量正是 `torch.cuda.memory_stats()` 里 `reserved_bytes.all.peak`、`segment.all.allocated`、`fragmentation_bytes` 的来源。与 `inf-paged-kv-block-alloc`（KV 页表分配，块大小恒定、无合并）和 `perf-oom-boundary-search`（解析式峰值公式）分工不同：本题是**动态块表的逐步演算**。

## 核心与易错点
1. **best-fit 不是 first-fit**：要在「装得下的块」里挑 `size` 最小的那个，并列再取 `start` 小的。first-fit 会先吃掉大块，把大块提前打碎，扩段次数显著变多。
2. **扩段后要重跑 best-fit**：新段加入空闲表后，最优块完全可能是更早释放出来的小块；直接在新段头部切一刀会让 reserved 白白变大。
3. **扩段量是 `max(need, grow_unit)` 且向上对齐**：申请超过 grow_unit 时按 need 扩（不浪费），小于时按 grow_unit 扩（少向设备要）。两条都要对齐到 `align`。
4. **释放后必须立刻与左右各合并一次**：漏掉任何一侧，空闲块表就不再是极大块集合，后续 best-fit 会看不出一整段连续空闲区，扩段次数暴涨——这是本题最典型的错误。
5. **切块后剩余仍空闲**：`size > need` 时剩余 `[start+need, start+size)` 要放回空闲表；忘了它就凭空吞掉显存。
6. **`peak_reserved` 只在实际扩段时更新**：reserved 单调不减，因此峰值一定出现在最后一次扩段之后；若在释放时误减 reserved，峰值和 final 都会错。

## 对拍口径
参考解维护按 `start` 有序的块表：`best_fit` 线性扫描取最小 size、`insert_free` 先并右侧再并左侧。判题 `_oracle` 完全不建块表，而是按 `align` 把地址切成布尔页位图：best-fit 由「扫描连续空闲页段、取满足 need 的最短段（并列取地址小者）」重新求出，极大块与合并由位图自然涌现，释放即把页翻回 True。两套结构（块表 vs 页位图）对全部 60 组随机操作序列逐项对拍；另设断言：单次大申请扩段量恰为 need、小申请扩段量恰为 grow_unit 且对齐、两次释放后合并成一块并原地复用不再扩段、全部释放后退化成 `[[0, reserved]]` 单块、空闲块两两不重叠且 start/size 都对齐、空操作序列返回全零。

## 复杂度
`O(P·B)`：`P` 为操作数、`B` 为空闲块数（块表线性扫描）。位图 oracle 为 `O(P·reserved/align)`，仅用于小规模对拍。
