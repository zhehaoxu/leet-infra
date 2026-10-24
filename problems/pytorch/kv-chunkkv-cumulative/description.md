# ChunkKV：语义块整体保留

## 题目描述

**ChunkKV**（Xu et al., 2025, [arXiv:2502.00297](https://arxiv.org/abs/2502.00297)）：跟 ChunkPress「每块内部各挑 token」不同，ChunkKV 关心的是**整块语义连贯性**——把一段连续 token 聚成一个 chunk、算出一个**块级分数**，然后按块级分数挑 top-k 个 chunk **整块保留 / 整块丢弃**，避免"块里挑几个 token 破坏语义"。

## 规则

给定每 head 的 token 分数 `scores`，形状 $(H, L)$；正整数 `chunk_length`；浮点 `compression_ratio` $\in[0,1)$：

1. 把长度维切成 $N=\lceil L/c\rceil$ 段（最后一段可能短于 $c$）。
2. **跨 head 求和、块内平均**得到块级分数：
   - 对每 token $t$ 先算 $s_t=\sum_h \text{scores}_{h,t}$（跨 head 求和）。
   - 块 $i$ 覆盖 $[ic,\min((i+1)c,L))$，$cs_i = \text{mean}_{t\in \text{seg}_i}(s_t)$（**除以块实际长度**）。
3. 保留块数 $n\_chunks\_kept = \max\big(1,\ \lfloor N \cdot (1-\text{compression\_ratio})\rfloor\big)$。
4. 按 $cs$ **降序**取 top $n\_chunks\_kept$（平局：块 index 小者优先）。
5. **所有被选中块的 token 位置合并**、**按位置升序**输出。块的选择跨 head 共享（因为步骤 2 已经求和掉 head 维）。

返回 shape $(n\_tokens\_kept,)$ 的一维 `long` 张量。

## 示例

```
H=2, L=12, chunk_length=4, compression_ratio=1/3
head0 scores: [0.5, 0.4, 0.6, 0.5, 0.05, 0.05, 0.05, 0.05, 0.4, 0.3, 0.5, 0.4]
head1 scores: [0.4, 0.5, 0.5, 0.4, 0.05, 0.05, 0.05, 0.05, 0.3, 0.4, 0.4, 0.3]
跨 head 求和: [0.9, 0.9, 1.1, 0.9 | 0.1, 0.1, 0.1, 0.1 | 0.7, 0.7, 0.9, 0.7]
块均分: cs = [0.95, 0.1, 0.75]
N=3, n_chunks_kept = max(1, floor(3·2/3)) = 2 → top2 = chunk0, chunk2
kept tokens = {0,1,2,3} ∪ {8,9,10,11} = [0,1,2,3,8,9,10,11]
```

## 约束

- CPU float32；块划分只沿 $L$ 维；$c\ge 1$。
- $N=1$（$L\le c$）时保底保留唯一 chunk → 全部位置。
- 平局按 $(-cs_i,\ i)$ 稳定排序。
