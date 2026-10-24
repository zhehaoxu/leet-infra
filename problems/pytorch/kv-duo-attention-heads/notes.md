# 解题说明 · DuoAttention

## 关键洞察：不是所有头都需要全量 KV

作者发现大模型注意力头高度分工：少数 **retrieval head** 承担"从很长上下文里回捞信息"的工作，它们的注意力会真的散布到很远的位置；而大量 **streaming head** 的注意力其实高度局部化——只看得到句首的 attention sink 和最近的窗口。既然如此，给后者留全量 KV 就是纯浪费。

DuoAttention 的压缩因此**发生在头的维度**：把 streaming head 降级成"只留 sink+recent"（本质是 per-head 的 StreamingLLM），retrieval head 保持全量。两者混着用，就得到"同等质量下更大的压缩比"。

## 头型如何判定

真实系统里头型是从数据集统计出来的（对每个头算"累积注意力曲线下面积"，越接近 1 越像 retrieval）。本题把这步**外置**成输入 `head_scores`——分数越高越"检索"，取最小的 $n\_pruned$ 个降级为流式头。这样题目聚焦在"分区掩码"这一步、不必复现统计流程。

## 与 StreamingLLM 的关系

单个流式头的掩码 `[T,T,F,F,F,F,T,T]` 正是 StreamingLLM 的 sink+window 结构。DuoAttention = "**按头**选择性地在某些层位置套用 StreamingLLM，其余头不动"。可以理解为：StreamingLLM 是"所有头都流式"，DuoAttention 是"只有被判定为流式的头才流式"。

## 平局与取整

- `n_pruned = floor(H·ratio + 0.5)` 明确半数向上、非银行家舍入。
- 头分平局时 index 小者优先进流式集——oracle 与实现都按 `(score, index)` 升序稳定排序，避免 `argsort` 平台差异导致 flaky。

## 窗口重叠的边界

当 `sink_size + recent_size >= seq_len` 时，流式头的首尾窗口把整行覆盖，掩码自然全 `True`。实现里用 `if L > S+R` 才去挖中间，`mask[h, S:L-R]` 在 $S+R\ge L$ 时是空切片，两种写法等价——本题保留显式守卫更直观。
