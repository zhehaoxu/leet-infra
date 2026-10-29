# 解题说明 · TTFT / TPOT 时间线

## 出处

- Orca: *A Distributed Serving System for Transformer-Based Generative Models*（OSDI '22）——提出 iteration-level scheduling（continuous batching）：每步重新组批，完成的请求立刻让位。
- Sarathi-Serve: *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*（arXiv:2403.02310）——chunked prefill 把长 prompt 切块与 decode 拼进同一步，用 prefill 分块大小调 TTFT 与 TPOT 的平衡。
- vLLM 等引擎把 TTFT / TPOT 作为标准 SLO 指标暴露；本题的"一步只做一件事"是最小化的共享 forward 模型。

## 核心与易错点

**prefill 和 decode 抢同一步。** 这是本题唯一不可省的物理约束：做 prefill 的那一步，所有在跑请求都不出词。示例 2 里 r0 的 3 个 token 间隔被 r1 的 4 个 prefill 步拉成 7 步，正是 chunked prefill 存在的理由；如果让 prefill 与 decode 同一步并行，TTFT 会好看但 TPOT 账就全错。

**准入在每步开头，完成的名额下步才可用。** 请求在第 t 步的 decode 里完成，名额释放发生在该步末尾；下一步准入才能用它。少写这一步顺序，所有请求会早一步首字。

**准入顺序是 `(arrival, 下标)`，不是数组顺序。** 到达时刻允许乱序，必须先排序；`arrival` 相同则按下标，保证确定性。

**prefill 是 FCFS 且一次只推一个请求。** 活跃集里"最早准入且没 prefill 完"的只有一个；不能把多个 prefill 拼在同一步，也不能让后来的请求插队。

**prefill 完成的那一步不出 token。** 首个 token 在其后的第一次 decode 步；`gen_len = 1` 的请求首 token 与末 token 同一步，`decode_span = 0`。

**空转要跳时钟。** 活跃集为空时不能一步步空转（`arrival` 可能很大），要把时钟直接推到下一个到达步，且空转不计入 `prefill_steps` / `decode_steps`。

**两个计数是不变量。** `prefill_steps` 恒等于 `Σ ceil(prompt_len_i / chunk)`（每个请求的 prefill 一定被走完）；`decode_steps` 在下界 `max(gen_len)` 与上界 `Σ gen_len` 之间。

## 对拍口径

- 参考实现：单个 `while` 循环 + 准入列表 + 字典记剩余 prefill，prefill / decode / 空转三分支各自 `continue`。
- 独立 oracle：`deque` 的 prefill 队列 + `list` 的 decode 集合，队列头部推进、完成即迁入 decode 集合；数据结构和控制流都与参考实现不同。
- 3000 组随机 `(arrivals, prompt_lens, gen_lens, chunk, max_batch)` 上两者五元组完全一致；另加手算时间线（示例 1–3）与不变量断言（`ttft >= ceil(prompt/chunk)`、`finish = arrival + ttft + span`、`prefill_steps` 恒等式）。

## 复杂度

- 每步 `O(活跃请求数)`，总步数不超过 `Σ ceil(prompt_len_i / chunk) + Σ gen_len_i`，整体 `O((Σ prefill 块数 + Σ gen_len) × n)`；`n <= 64` 时完全是毫秒级。

## 相关题目

- `inf-continuous-batching-scheduler`：只算完成步、不含 prefill 时间，是本题去掉 prefill 账的退化版。
- `inf-chunked-prefill-scheduling`：只按 token 预算切 chunk 大小，不做时间线。
- `inf-batch-spec-token-budget`：同一步内的名额分配，与本题的"步间时间线"互补。
