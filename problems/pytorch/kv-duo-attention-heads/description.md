# DuoAttention：检索头 vs 流式头的分区保留

## 题目描述

**DuoAttention**（Xiao et al., 2024, [arXiv:2410.10819](https://arxiv.org/abs/2410.10819)）的发现是：注意力头分两类——**retrieval head** 真的会回头检索很远处的 token（必须留全量 KV），而 **streaming head** 其实只关心"开头几个 + 最近几个"（留个 StreamingLLM 式的 sink+window 就够）。于是按重要性把**最不"检索"的一批头**降级为流式头、只留 sink+recent，其余头留全量——**头维度**上的混合压缩。

## 规则

给定 `head_scores` 形状 $(H,)$（每个头越像 retrieval head 分数越高）；`seq_len` $=L$、`sink_size` $=S$、`recent_size` $=R$、`head_compression_ratio` $\in[0,1]$：返回布尔保留掩码 `mask`，形状 $(H, L)$。

1. 转成流式头的数量 $n\_pruned = \lfloor H \cdot \text{head\_compression\_ratio} + 0.5\rfloor$（半数向上取整）。
2. 取 `head_scores` **最小** 的 $n\_pruned$ 个头作为**流式头**（平局：头 index 小者优先）。其余为 **retrieval 头**。
3. 逐头生成掩码：
   - **retrieval 头**：整行 `True`（全量保留）。
   - **流式头**：只保留开头 $S$ 个与结尾 $R$ 个，即位置 $t$ 满足 $t<S$ 或 $t\ge L-R$ 时为 `True`，中间段 `False`。

## 示例

```
H=5, L=8, S=2, R=2, head_compression_ratio=0.4
head_scores = [0.9, 0.2, 0.7, 0.1, 0.5]
n_pruned = round(5*0.4) = 2 → 最小的两个头 = {3(0.1), 1(0.2)} → 流式头 1、3

mask:
h=0 (retrieval): [T, T, T, T, T, T, T, T]
h=1 (streaming): [T, T, F, F, F, F, T, T]
h=2 (retrieval): [T, T, T, T, T, T, T, T]
h=3 (streaming): [T, T, F, F, F, F, T, T]
h=4 (retrieval): [T, T, T, T, T, T, T, T]
```

## 约束

- CPU；`mask` 为 `torch.bool`。
- 若 $S+R\ge L$，流式头也等价于整行 `True`（首尾窗口重叠覆盖全序列）。
- $H\cdot\text{ratio}$ 恰为 $.5$ 时**半数向上**取整（等价 $\lfloor x+0.5\rfloor$，非银行家舍入）。
- `head_scores` 长度即头数 $H$，无 batch 维。
