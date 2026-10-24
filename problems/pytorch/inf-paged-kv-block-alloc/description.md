# PagedAttention：KV Cache 分页 block 分配

## 题目描述

传统 KV cache 给每条序列预留一整块**连续**显存，序列长短不一导致大量内部碎片（预留了却没用满）与外部碎片（有空闲但接不上）。**PagedAttention**（vLLM）借鉴操作系统分页：把 KV cache 切成固定大小的 **block**（每块容纳 `block_size` 个 token 的 K/V），序列不再需要连续显存，而是持有一张 **block table**——把"逻辑第 $j$ 块"映射到任意"物理块号"。这样显存利用率接近 100%，也是 vLLM 高吞吐的根基。

本题模拟一次调度时的**准入分配**：物理块池 `free_blocks` 已按可用顺序给出，若干待跑序列按**请求编号顺序**（先来的先服务）依次申请，每条需要 $\lceil \text{len}/\text{block\_size}\rceil$ 个物理块。若剩余空闲不足以整条满足，则该序列**本步无法准入**（不部分分配），保持空表。

## 规则

给定 `free_blocks`（list[int]，物理块 id，从**头部**取用即 `pop(0)` 语义）、`block_size`（int）、`seq_lens`（一维，每条序列的 token 数，按请求 id 升序）：

1. 逐条序列 $s$（保持给定顺序）：需求 $\mathrm{need} = \lceil s/\text{block\_size}\rceil$，$s=0$ 时 $\mathrm{need}=0$
2. 若当前剩余空闲数 $\ge \mathrm{need}$：从空闲池**头部依次取出** `need` 个块 id 作为该序列的 block table，并把它们移出空闲池
3. 否则该序列 block table 记为 `[]`（未准入），**不占用**任何块
4. 处理完所有序列后，返回每条序列的 block table 与剩余空闲池

返回二元组 `(block_tables, remaining_free)`：`block_tables` 为 `list[list[int]]`（长度 = 序列数，未准入为 `[]`），`remaining_free` 为 `list[int]`（尚未被占用的物理块，保持原相对顺序）。

## 示例

```
free_blocks = [0,1,2,3,4,5], block_size = 4
seq_lens    = [10, 4, 6, 5]   → need = [3, 1, 2, 2]  (共需 8 > 6)

s=10: need=3 → [0,1,2]，free=[3,4,5]
s=4:  need=1 → [3]，      free=[4,5]
s=6:  need=2 → [4,5]，    free=[]
s=5:  need=2 → 空闲不足 → []（未准入）

block_tables    = [[0,1,2],[3],[4,5],[]]
remaining_free  = []
```

## 约束

- CPU；`block_size >= 1`；物理块 id 互不相同。
- 分配严格按 `seq_lens` 给定顺序、每序列从空闲池头部取块。
- 未准入序列用空列表 `[]` 表示（不做部分分配、不阻塞其后仍能准入的短序列）。
