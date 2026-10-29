# 连续批处理的 TTFT / TPOT 时间线：prefill 与 decode 共享同一步

## 题目描述

在线推理服务有两个最常被写进 SLO 的指标：

- **TTFT**（time to first token）：请求到达到吐出第一个 token 的延迟，决定用户"感觉快不快"；
- **TPOT / ITL**（time per output token / inter-token latency）：首 token 之后相邻两个 token 的间隔，决定"读起来顺不顺"。

这两个指标不是算力问题，而是**调度**问题。连续批处理（continuous batching）把不同请求的 decode 拼进同一次 forward，吞吐上去了；但 prefill 与 decode 抢的是同一次 forward —— 每做一次 prefill 步，所有正在 decode 的请求这一步就出不了词。于是新请求的 prefill 会把老请求的 TPOT 拉长，这就是 chunked prefill 要平衡的那笔账。

本题用整数步模拟一个最小的 serving 引擎：给定到达时刻、prompt 长度、生成长度、prefill 分块大小与批上限，输出每个请求的 TTFT、decode 跨度与完成步，以及全过程的 prefill / decode 步数。

## 规则

时钟以整数步计，第 0 步是第一个时间点。**活跃请求** = 已准入且尚未完成的请求（prefill 中和 decode 中都各占一个批名额）。

每一步的流程：

1. **准入**：只要活跃请求数小于 `max_batch`，就按 `(arrivals, 下标)` 升序把已到达的等待请求逐个放进活跃集，直到满员或没有已到达的请求。本步新准入的请求可以立刻参与本步动作。
2. **选动作**（一步只做一种）：
   - **prefill 步**：活跃集里还有 prompt 没处理完的请求时，取最早准入（即 FCFS）的那个，推进 `min(chunk, 剩余)` 个 prompt token。本步不做 decode，所有在跑请求都不出词；
   - **decode 步**：否则，若活跃集里有已 prefill 完的请求，所有已 prefill 完且未完成的活跃请求各产出 1 个 token。恰好产满 `gen_lens[i]` 个的请求在本步完成并让出名额；本步让出的名额要等下一步准入才能使用；
   - **空转**：以上都没有时，时钟直接跳到下一个到达步，空转不计入任何步数统计。

prefill 完成的那一步**不**产出 token；第一个 token 出现在其后的第一次 decode 步。

返回五元组：

- `ttft`：每个请求的「首个 token 步 − 到达步」；
- `decode_span`：每个请求的「末个 token 步 − 首个 token 步」，`gen_lens[i] == 1` 时为 0；
- `finish`：每个请求末个 token 的绝对步号；
- `prefill_steps`：全过程做 prefill 步的次数；
- `decode_steps`：全过程做 decode 步的次数。

## 示例

**示例 1**：`arrivals = [0, 0, 2]`、`prompt_lens = [8, 4, 4]`、`gen_lens = [2, 3, 1]`、`chunk = 4`、`max_batch = 2`

```
步 0  准入 r0、r1（r2 未到）        prefill r0: 8 -> 4
步 1  prefill r0: 4 -> 0（完成）
步 2  prefill r1: 4 -> 0（完成）    （r2 已在步 2 到达，但名额已满）
步 3  decode: r0 出第 1 个、r1 出第 1 个   <- 两者的首个 token
步 4  decode: r0 出第 2 个 -> 完成；r1 出第 2 个
步 5  准入 r2（接 r0 让出的名额）    prefill r2: 4 -> 0
步 6  decode: r1 出第 3 个 -> 完成；r2 出第 1 个 -> 完成（gen_len = 1）
```

```
ttft        = [3, 3, 4]     # r2 到步 2 才到，又等到步 5 才准入
decode_span = [1, 3, 0]
finish      = [4, 6, 6]
prefill_steps = 4
decode_steps  = 3
```

**示例 2**（prefill 挤占 decode）：`arrivals = [0, 2]`、`prompt_lens = [1, 8]`、`gen_lens = [4, 1]`、`chunk = 2`、`max_batch = 2`

```
步 0  准入 r0                 prefill r0: 1 -> 0
步 1  decode r0 第 1 个        <- r0 的 TTFT 只有 1 步
步 2  准入 r1                 prefill r1: 8 -> 6
步 3  prefill r1: 6 -> 4
步 4  prefill r1: 4 -> 2
步 5  prefill r1: 2 -> 0（完成）
步 6  decode r0 第 2 个、r1 第 1 个
步 7  decode r0 第 3 个
步 8  decode r0 第 4 个 -> 完成
```

```
ttft        = [1, 4]
decode_span = [7, 0]        # r0 的 3 个间隔平均 7/3 步，而非 1 步
finish      = [8, 6]
prefill_steps = 5
decode_steps  = 4
```

**示例 3**（名额一个、严格串行）：`arrivals = [0, 0, 0]`、`prompt_lens = [1, 1, 1]`、`gen_lens = [1, 1, 1]`、`chunk = 1`、`max_batch = 1`

```
ttft        = [1, 3, 5]
decode_span = [0, 0, 0]
finish      = [1, 3, 5]
prefill_steps = 3
decode_steps  = 3
```

## 约束

- 请求数 `n` 不超过 64；`arrivals[i]` 非负且允许重复与乱序；`prompt_lens[i] >= 1`；`gen_lens[i] >= 1`；`chunk >= 1`；`max_batch >= 1`。
- 所有请求最终都会完成（`max_batch >= 1` 保证等待队列不会永久堵死）。
- 返回的两个列表与两个计数都是 Python `int`；`finish` 是绝对步号（不是相对到达的延迟）。
- 判题只看数值，不限定模拟写法（字典 + 单循环、事件队列、时间轮都可以）。
