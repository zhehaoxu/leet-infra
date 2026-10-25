# 解码 Roofline 的 Batch 账：KV 读与算力拐点

## 题目描述

解码（decode）每生成一个 token 都要把**全部权重**从 HBM 读一遍，但这份读量与 batch 无关；同时每条序列还要
把自己的 **KV cache** 读一遍，这份读量**严格正比于 batch**。于是小 batch 时权重读量占主导、加 batch 几乎
线性提速，而 batch 大到 KV 反超权重后，**每 token 的 HBM 读量不再下降**，吞吐就撞在 $bw/K$ 这堵墙上——
哪怕算力还没用满。把这条曲线算清楚，就知道「该堆多大的 batch」「显存里的 KV 到底值不值」「长上下文服务
为什么吞吐上不去」。

`roofline-arithmetic` 算的是单个算子的性能上限，`inference-throughput` 只按算力折算吞吐（不看 HBM 读），
`kv-cache-memory` 只给 KV 占多少字节；本题把两者合到**一个随 batch 变化的时间轴**上：每步搬多少字节、
每步算多少 FLOP、拐点在哪、SLO 要多大 batch。

## 规则

- 每步从 HBM 搬的字节 = **权重读一遍** + **每条序列的 KV 各读一遍**；KV 的每 token 字节沿用
  `kv-cache-memory` 的 $2\cdot L\cdot n_{kv}\cdot d\cdot T$（2 是 K 与 V），再乘 `kv_bytes`。
- 每步的 FLOP 用 `inference-throughput` 的 $2N$ 口径：$\text{flops\_per\_step}=B\cdot 2N$。
- **单步时间取两者较大**：$\text{step\_time}=\max(\text{bytes}/bw,\ \text{flops}/peak)$，
  $\text{tokens\_per\_s}=B/\text{step\_time}$（ms 口径只用于 `step_time_ms`）。
- **瓶颈判定**沿用 `roofline-arithmetic`：算强度不低于 ridge（即 $\text{flops}\ge \text{bytes}\cdot peak/bw$）
  记 compute-bound，**相等也算**，所以 `is_memory_bound` 用严格小于。
- **上限** $\text{ceiling}=\min(peak/(2N),\ bw/K)$：前者是算力上限，后者是 $B\to\infty$ 时 KV 读的上限
  （$K=0$ 时只有前者）。它同时也是「加 batch 换不来收益」的饱和点。
- **拐点** `balance_batch` 是进入 compute-bound 的**最小正整数** $B$，由
  $B\cdot(2N\cdot bw-K\cdot peak)\ge peak\cdot W$ 解出（$W$ 是权重字节）；系数 $\le 0$ 说明
  「KV 读永远压不住」，记 $-1$，此时吞吐只趋近 $bw/K$。
- **SLO 反解** `min_batch_for_slo`：在 $[1, \texttt{max\_batch}]$ 内使 $\text{tokens\_per\_s}\ge$
  `target_tokens_per_s` 的最小 $B$（该函数对 $B$ 单调不减，可二分），找不到记 $-1$；
  `target_tokens_per_s=0` 时答案恒为 1。
- 输入含义、每条公式与返回类型口径全部写在 `starter.py` 的 docstring 里，以它为准。

## 示例

```
输入：n_params=6.6e9, layers=32, kv_heads=8, head_dim=128, context_tokens=2048, batch=32,
      peak_flops=312e12, bandwidth=2e12, target_tokens_per_s=2000（A100 上的 7B + GQA）
  权重读 = 6.6e9*2 = 13,200,000,000 字节；每序列 KV = 2*32*8*128*2048*2 = 268,435,456 字节
  bytes/step = 13.2e9 + 32*268,435,456 = 21,789,934,592   flops/step = 32*2*6.6e9 = 422,400,000,000
  step_time = max(21,789,934,592/2e12, 422.4e9/312e12) = 0.010894967296 s -> 10.894967296 ms
  tokens_per_s = 32/0.010894967296 = 2937.135939063217（memory-bound：422.4e9 < 21.79e9*156）
  ceiling = min(312e12/13.2e9, 2e12/268,435,456) = 7450.580596923828   <- 被 KV 读钉死
  2N*bw - K*peak = 2.64e22 - 8.375e22 < 0 -> balance_batch = -1（加多大 batch 都填不满算力）
  min_batch_for_slo(2000) = 19   weight_fraction = 13.2e9/21.79e9 = 0.6057842874317886
输出：(2937.135939063217, True, 10.894967296, 7450.580596923828, -1, 19, 0.6057842874317886, 13200000000)

输入：同上但 context_tokens=0（没有 KV，只剩权重读），target_tokens_per_s=2000
  bytes/step = 13.2e9（权重）  step_time = max(13.2e9/2e12, 422.4e9/312e12) = 6.6 ms
  tokens_per_s = 32/0.0066 = 4848.484848484849（仍是 memory-bound）
  ceiling = 312e12/13.2e9 = 23636.363636363636   balance_batch = 312e12*13.2e9/(13.2e9*2e12) = 156
  min_batch_for_slo(2000) = 14（每加一条序列多 151.5 tokens/s）  weight_fraction = 1.0
输出：(4848.484848484849, True, 6.6, 23636.363636363636, 156, 14, 1.0, 13200000000)

输入：n_params=1000, layers=2, kv_heads=2, head_dim=8, context_tokens=0, batch=1,
      peak_flops=1e9, bandwidth=1e9（刻意造出「刚好相等」的边界）
  bytes = 2000、flops = 2000、ridge = 1 -> flops == bytes*ridge，相等记 compute-bound
  step_time = 2e-6 s -> 0.002 ms，tokens_per_s = 500000.0 == ceiling，balance_batch=1
输出：(500000.0, False, 0.002, 500000.0, 1, 1, 1.0, 2000)
```

## 约束

- `n_params` $\ge 1$；`layers`、`kv_heads`、`head_dim` $\ge 1$；`context_tokens` $\ge 0$；`batch` $\ge 1$；
  `peak_flops`、`bandwidth` $>0$；`weight_bytes`、`kv_bytes` $\ge 1$；`target_tokens_per_s` $\ge 0$；
  `max_batch` $\ge 1$
- 返回八元组 `(tokens_per_s, is_memory_bound, step_time_ms, ceiling_tokens_per_s, balance_batch,
  min_batch_for_slo, weight_fraction, flops_per_token)`：第 1、3、4、7 项 float（`atol=1e-5, rtol=1e-5`），
  第 2 项 bool，第 5、6、8 项 int（精确相等）
- 只建模 HBM 读（权重 + KV）与稠密算力，不考虑 prefill、调度开销、显存容量能否放下这么大 batch；
  `balance_batch` 是物理拐点，不受 `max_batch` 截断
