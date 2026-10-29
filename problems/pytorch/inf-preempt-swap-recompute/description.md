# 抢占决策：换出还是重算

## 题目描述

在线服务的 KV 缓存池是固定大小的。当新请求的 prefill 需要腾地方、而已经没有空闲槽位时，引擎必须在在跑请求里挑牺牲者（preemption）。挑中之后还有第二种选择：

- **swap（换出）**：把该请求的 KV 搬到主机内存，腾出显存；等资源宽松时再搬回来继续解码。代价是两次搬运 `2 × kv_tokens × bytes_per_token / 带宽`。
- **recompute（重算）**：直接丢掉 KV，将来重新把这个请求整段 prefill 一遍（prompt + 已生成的 token 全都要重跑）。代价是 `(prompt_len + generated) × 每 token 重算拍数`。

序列短、生成少时重算更便宜；序列长或已经生成很多时换出更划算。本题给定当前在跑请求的账本，输出该抢谁、各自走哪条路、腾出多少槽位、总开销多少拍。

## 规则

1. **谁能被抢**：只有 `freezable[i]` 为 `True` 的请求进入候选集，其余是受保护的（例如正在收尾或高 SLA 请求），无论腾出多少都不动。
2. **按优先级分层**：候选按 `priority` 从低到高逐层处理；**同一层内**先挑 `kv` 更大的（一次腾出更多），`kv` 相同则下标小的先挑。必须先把低优先级层试完，才允许升到下一层。
3. **够用即停**：一旦累计腾出的槽位 `>= need`，立刻停止，不去「顺手多腾一点」。
4. **逐个比较**：
   - `swap_ticks = ceil(2 * kv[i] * bytes_per_token / swap_bw)`
   - `recompute_ticks = (prompt[i] + generated[i]) * tick_per_token`
   - `swap_ticks <= recompute_ticks` 时选 `"swap"`，否则选 `"recompute"`。
5. **不可行**：把所有候选都抢完仍然不够 `need`，则整体放弃：不做任何抢占，`ticks` 记为 `-1`、`replay_tokens` 记为 0、`victims` 与 `modes` 为空列表，但 `freed` 仍如实报告「可腾出的总量」。

返回五元组：

- `victims`：被抢占请求的下标，按选择先后排列；
- `modes`：与 `victims` 一一对应的 `"swap"` / `"recompute"`；
- `freed`：实际腾出的槽位数（允许大于 `need`）；
- `ticks`：所有被抢占请求的开销之和；不可行时为 `-1`；
- `replay_tokens`：走 `recompute` 的请求需要重新 prefill 的 token 总数；不可行时为 0。

## 示例

**示例 1**（同一层内两种模式各出现一次）：`kv = [4, 40, 40]`、`prompt = [30, 5, 5]`、`generated = [10, 100, 1]`、`priority = [0, 0, 0]`、`need = 44`、`bytes_per_token = 4`、`swap_bw = 16`、`tick_per_token = 1`、全部可抢占

```
候选按 kv 降序：1(40)、2(40)、0(4)
请求 1：swap = ceil(2*40*4/16) = 20，重算 = (5+100)*1 = 105  -> swap，20 拍
        已腾 40 < 44，继续
请求 2：swap = 20，重算 = (5+1)*1 = 6                       -> recompute，6 拍，重放 6 个 token
        已腾 80 >= 44，停止（请求 0 不动）
```

```
victims       = [1, 2]
modes         = ["swap", "recompute"]
freed         = 80
ticks         = 26
replay_tokens = 6
```

**示例 2**（低优先级层不够才升级）：`kv = [30, 6, 50]`、`prompt = [10, 10, 10]`、`generated = [2, 2, 2]`、`priority = [2, 0, 1]`、`need = 20`、`bytes_per_token = 4`、`swap_bw = 64`、`tick_per_token = 1`

```
优先级 0 层：只有请求 1(kv=6) -> swap，ceil(48/64)=1 拍，已腾 6 < 20
优先级 1 层：请求 2(kv=50)    -> swap，ceil(400/64)=7 拍，已腾 56 >= 20，停止
```

```
victims = [1, 2]
modes   = ["swap", "swap"]
freed   = 56
ticks   = 8
replay_tokens = 0
```

**示例 3**（受保护请求不参与）：`kv = [30, 30, 10]`、`prompt = [8, 8, 4]`、`generated = [2, 2, 1]`、`priority = [0, 1, 2]`、`need = 20`、`bytes_per_token = 4`、`swap_bw = 64`、`tick_per_token = 1`、`freezable = [True, False, False]`

```
只有请求 0 可抢占：swap = ceil(2*30*4/64) = 4 拍，腾出 30 >= 20
```

```
victims = [0]
modes   = ["swap"]
freed   = 30
ticks   = 4
replay_tokens = 0
```

## 约束

- `n = len(kv) = len(prompt) = len(generated) = len(priority) = len(freezable)`，`1 <= n <= 64`。
- `kv[i] >= 1`、`prompt[i] >= 1`、`generated[i] >= 0`、`1 <= need`、`bytes_per_token >= 1`、`swap_bw >= 1`、`tick_per_token >= 1`。
- 所有计算都是整数：`ceil` 用整数上取整，不要用浮点除法再取整。
- 判题只看数值与列表内容，不限定实现方式。
