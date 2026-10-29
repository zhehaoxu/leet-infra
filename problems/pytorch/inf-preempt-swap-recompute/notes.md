# 解题说明 · 抢占决策：换出还是重算

## 出处

- vLLM 的调度器把「显存不足 → 抢占」做成两条路径：默认 `RECOMPUTE`（丢掉 KV，之后重新 prefill），可配置为 `SWAP`（KV 搬到 CPU 再搬回）；牺牲者取自运行队列尾部、按请求优先级挑选。见 `vllm/core/scheduler.py` 的 `_preempt` 与 `PreemptionMode`。
- Orca（OSDI '22）的 iteration-level scheduling 与 Sarathi-Serve（arXiv:2403.02310）的 chunked prefill 是「抢占后如何恢复」的背景：被重算的请求要重新吃一遍 prefill 算力。
- 「换出 vs 重算」的量化比较是这两条实现路径的抽象：搬运成本只与 KV 字节数有关，重算成本与整段序列长度有关。

## 核心与易错点

**重算的代价是 `prompt + generated`，不是 `prompt`。** 已经生成的 token 也要重新跑一遍才能回到原来的状态；漏掉 `generated` 会让长输出请求被错误地判成「重算便宜」。

**换出要算两趟。** 搬出去、将来还要搬回来，成本是 `2 × kv × bytes_per_token / 带宽`；只算一趟会让 swap 看起来便宜一倍。

**ceil 必须用整数。** `ceil(2*kv*bpt/bw)` 用 `-(-a // b)` 或 `(a + b - 1) // b`，浮点除法在 `2*kv*bpt/bw` 恰好略大于整数时可能取整错位。

**先分层，再比大小。** 顺序是「优先级升序 → 同层 kv 降序 → 下标升序」的字典序，不是全局按 kv 排序。示例 2 里 `kv = 6` 的请求先于 `kv = 50` 的被抢，正因为它处在更低的优先级层。

**够用即停。** 腾出量 `freed` 可以大于 `need`（示例 1 是 80 > 44），但不能为了「凑整」多抢一个。

**不可行要整体放弃。** 抢完所有候选仍不够时，`victims`/`modes` 为空、`ticks = -1`、`replay_tokens = 0`，但 `freed` 仍报告「本来能腾出多少」——这个字段是给调度器做失败诊断用的，不要一起清成 0。

**打平时选 swap。** `swap_ticks <= recompute_ticks` 取 swap，边界用例（`2*kv*bpt/bw` 恰好等于重算拍数）必须走 swap 分支。

## 对拍口径

- 参考实现：`sorted({...})` 求优先级层，层内 `candidates.sort(key=(-kv, index))`，主循环里 `if freed >= need: break` 双重短路。
- 独立 oracle：用 `dict` 分桶（`setdefault(level, []).append(i)`），桶内直接 `sorted(key=(kv, -i), reverse=True)`，代价用 `math.ceil` 而不是整数上取整——两种写法的排序与取整路径都不同。
- 4000 组随机实例（`n ≤ 6`、kv ≤ 20、带宽/拍数随机、约 20% 受保护）上五元组逐位一致；另有 9 个手算场景（示例 1–3、swap/recompute 各自的便宜侧、打平、同层同 kv、够用即停、不可行、受保护）与随机不变量（`freed == Σ kv[victims]`、victims 的优先级非降、`ticks >= replay_tokens * tick_per_token`）。

## 复杂度

- 分桶 + 排序 `O(n log n)`，主循环 `O(n)`；`n <= 64`，常数级。
- 真实引擎里这一步还要算 KV 块页表的搬移与异步拷贝句柄，本题只保留时间账。

## 相关题目

- `inf-kv-block-admission`：按块预算决定「能不能准入」，本题决定「准入不进时抢谁」。
- `inf-lru-kv-cache-eviction`：缓存层面的 LRU 驱逐，牺牲者按最近使用时刻选，不考虑 swap/recompute 成本。
- `inf-edf-deadline-scheduler`：按截止期排序的调度，可与本题的优先级分层对照。
- `perf-oom-boundary-search`：训练侧的显存边界搜索，是「先算出需要多少再决定」的另一半。
