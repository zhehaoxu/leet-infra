# ChunkPress：分块独立 Top-K 保留

## 题目描述

**ChunkPress**（源于 FINCH，Zhu et al., 2024, [arXiv:2407.09112](https://arxiv.org/abs/2407.09112)）是一个"包装器"式压缩法：普通的 ScorerPress 会在全序列上做一次性 top-k，容易把保留名额都堆到少数高分区段。ChunkPress 把序列切成**等长 chunk**，在**每个 chunk 内部各自 top-k**，从而保证压缩在整个上下文里均匀铺开、不会漏掉任何区段。

## 规则

给定每个 head 的 token 重要性分数 `scores`，形状 $(H, L)$；正整数 `chunk_length`；浮点 `compression_ratio` $\in[0,1)$：

1. 沿长度维把 $L$ 切成若干段：第 $i$ 段覆盖位置 $[ic,\ \min((i+1)c, L))$，其中 $c=$ `chunk_length`。**最后一段可能不足 $c**。
2. 对每一段，保留数量
   $$n\_kept = \max\big(1,\ \lfloor \text{len}_{seg}\cdot(1-\text{compression\_ratio})\rfloor\big)$$
   其中 $\text{len}_{seg}$ 是该段的**实际长度**。
3. 在该段内按分数取 top $n\_kept$（分数大者优先），返回这些位置的**全局下标**（即段内相对下标 + 段起始偏移）。
4. 各段结果**按段顺序拼接**；同一段内**按分数从高到低**排列。

输出 shape $(H, n\_total)$ 的 `long` 张量，$n\_total$ 为各段保留数之和。

## 示例

```
H=2, L=8, chunk_length=4, compression_ratio=0.5
head 0 scores: [0.1, 0.9, 0.3, 0.2 | 0.8, 0.05, 0.7, 0.6]
  chunk0 len=4, n_kept=2, top2 = 位置1(0.9), 位置2(0.3)
  chunk1 len=4, n_kept=2, top2 = 位置4(0.8), 位置6(0.7)
head 1 scores: [0.5, 0.4, 0.9, 0.1 | 0.05, 0.7, 0.6, 0.8]
  chunk0 top2 = 位置2(0.9), 位置0(0.5)
  chunk1 top2 = 位置7(0.8), 位置5(0.7)

out = [[1, 2, 4, 6],
       [2, 0, 7, 5]]
```

## 约束

- CPU float32；`scores` 每行独立处理，不做跨 head 汇聚。
- `compression_ratio` $=0$ 时每段保留整段；$\to 1$ 时每段保底 1 个。
- 平局：分数相同则**位置小者优先**（stable，按 $(−\text{score},\ \text{pos})$ 排序）。
