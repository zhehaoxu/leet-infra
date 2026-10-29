# 解题说明 · 每步 token 预算背包

## 出处

- Sarathi-Serve: *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*（arXiv:2403.02310）——stall-free batching + chunked prefill：decode 保底占名额，剩余预算给 prefill 分块，避免长 prompt 把在跑请求饿死。
- Orca（OSDI '22）的 iteration-level scheduling 与 vLLM 的连续批处理是这套预算模型的前身；生产引擎（vLLM、DistServe 等）在此之上再加优先级/SLO 权重，就是本题的 `values`。
- 把"这一步准入哪些请求"写成 0/1 背包，是这类调度论文里最常用的抽象；真实系统会用最大权重的近似或贪心，本题要求精确最优。

## 核心与易错点

**先扣 decode，再谈 prefill。** `remaining = budget - decode_count`，decode 是保底负载，不能为了 prefill 而少服务 decode（stall-free 的前提）。忘扣这一步，所有入选集合都会多塞进一两个请求。

**目标是三个关键字的字典序，不是单一目标。** 权重最大 → 占用最小 → 下标列表字典序最小。第二关键字很关键：本轮省下的名额会变成下一轮的分块预算，示例 2 里 `{0,2}` 与 `{0,1,2}` 权重相同，占用小的那个才是对的。

**0 权重请求永远不入选。** 它只会消耗名额而不增加价值；第三关键字（下标字典序）会让"什么都不选"胜出。但它仍可能拿到余量分块，别把它从分块候选里排除。

**余量给 FCFS 队头，不是给最便宜的。** 入选集合是按背包最优选的，余量分块却是队列语义：下标最小的未入选请求先走一步。示例 3 里入选的是下标 0，分块却给了下标 1。

**只分一块，不级联。** 分块之后剩下的 token 记为 `unused`，不会再喂给下一个请求；这对应"一步只推进一个 prefill 分块"的引擎行为。

**n 很小，但别退化成枚举。** 题目约定 `n <= 12`，`2^n` 枚举能过；但容量 DP `O(n · remaining)` 才是标准解法，且天然支持更大的队列。

## 对拍口径

- 参考实现：按容量索引的 0/1 背包 DP，每个容量保存 `(权重, 升序下标元组)`，同容量上用 `(-权重, 下标元组)` 比较；最后在所有 `capacity <= remaining` 的状态里按 `(-权重, 容量, 下标元组)` 取最优。
- 独立 oracle：枚举全部 `2^n` 个子集，按 `(-权重和, 占用, 下标元组)` 取最优。
- 4000 组随机 `(prompt_lens, values, budget, decode_count)` 上两者五元组完全一致（含 0 权重、空队列、无解等退化情形）；另加逐项不变量：入选集合不超预算、分块不落在入选请求上、分块下标必须是未入选里的最小下标、`chunk_tokens + unused` 与余量守恒。

## 复杂度

- DP：时间 `O(n · remaining)`，空间 `O(remaining)`；`n <= 12`、`remaining <= 128` 时是瞬间完成，`2^n` 的暴力枚举同样可接受。

## 相关题目

- `inf-chunked-prefill-scheduling`：只按预算切 chunk 大小，不做"准入谁"的取舍。
- `inf-batch-spec-token-budget`：同一步内的名额分配，但收益函数是凹的、用贪心即可，不需要背包。
- `inf-ttft-tpot-queue`：把每步的决策串成时间线，看 TTFT/TPOT 怎么被 prefill 挤占。
