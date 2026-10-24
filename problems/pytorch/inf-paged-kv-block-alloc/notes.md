# 解题说明 · PagedAttention block 分配

## 分块向上取整

序列长度 `len` 落在 $[\,(j-1)b,\; jb\,]$ 之间就需要 $j$ 个块，即 $\lceil \text{len}/b\rceil$。最后一块通常只填一部分（内部碎片被压到 $<b$ 个 token），这正是分页相比"整段连续预留"的省处。整数写法 `-(-len//b)` 避免浮点误差。

## 准入顺序 = 先到先得

真实 vLLM 里 block 分配发生在每次 decode step 追加 token 时（写满当前块才 `allocate` 新块），并且会配合抢占（preemption）把老请求的块回收再让给新请求。本题把它简化成"一批序列一次性准入"，严格按请求编号顺序贪心取块：只要空闲够就分配，不够就整条拒（不部分分配），因此更靠后的短序列仍可能在长序列失败后成功准入。

## block table 的意义

`block_tables[i]` 就是序列 $i$ 的**页表**：逻辑块 $j$（第 $j\cdot b$ 到第 $(j+1)b$ 个 token 的 KV）→ 物理块号 `block_tables[i][j]`。注意力 kernel 读 KV 时先查页表再定位显存，从而支持非连续存储与跨请求共享（prefix caching 就是让多条页表指向同一物理块）。
