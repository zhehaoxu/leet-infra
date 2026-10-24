# StreamingLLM：Sink + 滑窗 KV 打分

## 题目描述

StreamingLLM（Xiao et al., ICLR 2024）观察：LLM 会把大量注意力打在开头几个 token 上——称为 **attention sink**。因此 KV 缓存可以只保 "**sink + 最近窗口**"，中间整段直接扔。

kvpress 里的 `StreamingLLMPress` 用一个 **二值打分向量** 表达这个策略：所有位置初始化为 1，把要被剪掉的**中段**连续一段设为 0，然后交给 `ScorerPress` 的 `topk(n_kept)` 挑保留位。

给定缓存长度 `k_len`、sink 数 `n_sink` 与压缩比 `compression_ratio ∈ [0, 1)`，先算 `n_pruned = k_len - int(k_len * (1 - compression_ratio))`，然后返回长度为 `k_len` 的 float 分数向量：位置 `[n_sink, n_sink + n_pruned)` 置 **0**，其他位置置 **1**。

## 规则

实现 `solve(k_len, n_sink, compression_ratio)` → 返回 shape `(k_len,)` 的 float32 tensor。

## 示例

```
k_len = 10, n_sink = 2, compression_ratio = 0.4
n_pruned = 10 - int(10 * 0.6) = 10 - 6 = 4
scores = [1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
        └ sink ┘ └ pruned ┘ └ window ┘
```

## 约束

- CPU float32；`int(k_len * (1 - cr))` 用 Python/torch 的向零截断（不是 round）。
- `n_sink + n_pruned <= k_len`（题目输入保证）。
