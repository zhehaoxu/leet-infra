# PD 分离的放置账：KV 跨节点搬运、TP 环形复制与显存准入

## 题目描述

把 prefill 和 decode 拆到不同实例上（disaggregation）之后，服务端多了一笔过去不存在的账：**prefill 算完的 KV 要搬到某个 decode 副本上**。这笔账同时受三件事约束——prefill 池什么时候算完、KV 传输链路什么时候空出来、目标副本还剩多少显存。

本题给定一批请求、若干 decode 副本（各自容量、链路带宽、张量并行度）和一个 prefill 池吞吐，要求模拟一遍放置过程：每个请求落到哪个副本、KV 全部就位是哪一拍、每条链路累计搬了多少字节、每个副本占了多少显存，以及有多少请求因为哪个副本都装不下而被丢弃。

服务拓扑与假设：

- 所有请求在下标顺序上依次做 prefill（FCFS、互不重叠），prefill 池吞吐为 `prefill_tps` 个 prompt token/拍。
- prefill 完成后，KV 通过该副本的传输链路搬过去；**搬运与 prefill 计算不重叠**，但每条链路同一时刻只搬一个请求，后来者要排队。
- decode 副本若要并行，KV 还需要在副本内部做一次环形 all-gather（让每个 TP rank 都拿到完整 KV），这笔流量按「每条链路分到 (C−1)/C 份额」记在同一条链路上。
- 请求的显存占用按峰值算：prompt 的 KV 加上将要生成的 token 的 KV。

## 规则

1. **prefill 完成拍**：记 `cum_prompt[i]` 为 `prompt_lens[0..i]` 之和，则

   ```
   prefill_done[i] = ceil(cum_prompt[i] / prefill_tps)
   ```

   用**累计量整体上取整**，不是每个请求各自取整。

2. **请求的 KV 字节数**（峰值口径）：

   ```
   kv[i] = (prompt_lens[i] + gen_lens[i]) * kv_bytes_per_token
   ```

3. **一条链路上要搬多少**：对 decode 副本 `d`，

   ```
   ring   = ceil(kv[i] * (tps[d] - 1) / tps[d])      # 环形 all-gather 的链路份额
   moved  = kv[i] + ring                              # tps[d] == 1 时 ring == 0
   ```

4. **逐请求放置**（按下标从小到大，决定即生效）：

   - 先做显存准入：只保留满足 `resident[d] + kv[i] <= caps[d]` 的副本（等号算装得下）。
   - 对每个可用副本预估完成拍：

     ```
     start  = max(prefill_done[i], link_free[d])
     finish = start + ceil(moved / bws[d])
     ```

     其中 `link_free[d]` 是该链路上一个传输的完成拍（初始为 0）。
   - 取 `finish` 最小的副本；`finish` 相同则取下标小的。注意**不是**挑最闲或带宽最大的副本。
   - 放置成功后：`resident[d] += kv[i]`、`wire[d] += moved`、`link_free[d] = finish`。

5. **丢弃**：一个副本都不满足准入时，该请求被 drop——`placement` 与 `ttft` 记 `-1`，不占显存也不占链路；但它的 prefill 份额**已经消耗**，后续请求的 `prefill_done` 仍按包含它的累计量计算。

6. **TTFT** 定义为「KV 全部就位的那一拍」，也就是该请求首个 token 可以出的时刻。

返回六元组：

- `placement`：每个请求落到哪个 decode 副本；被拒绝为 `-1`；
- `ttft`：每个请求的 KV 就位拍；被拒绝为 `-1`；
- `wire`：每个副本链路的累计搬运字节数（含环形份额）；
- `resident`：每个副本上被接纳请求的 KV 字节总数；
- `rejected`：被拒绝的请求数；
- `makespan`：所有被接纳请求 `ttft` 的最大值；没有请求被接纳时为 0。

## 示例

`prompt_lens = [20, 30, 100, 10]`、`gen_lens = [30, 10, 100, 5]`、`kv_bytes_per_token = 4`、`caps = [1000, 1000]`、`bws = [64, 32]`、`tps = [1, 2]`、`prefill_tps = 10`

各请求 prefill 完成拍：累计 20/50/150/160 → `2 / 5 / 15 / 16`。

```
请求 0：kv = (20+30)*4 = 200
  副本 0(tp=1)：moved = 200，start = 2，finish = 2 + ceil(200/64) = 6
  副本 1(tp=2)：ring = ceil(200*1/2) = 100，moved = 300，finish = 2 + ceil(300/32) = 12
  -> 副本 0，ttft = 6；resident[0] = 200

请求 1：kv = (30+10)*4 = 160
  副本 0：start = max(5, 6) = 6，finish = 6 + ceil(160/64) = 9
  副本 1：start = 5，moved = 240，finish = 5 + ceil(240/32) = 13
  -> 副本 0，ttft = 9；resident[0] = 360

请求 2：kv = (100+100)*4 = 800
  副本 0：360 + 800 = 1160 > 1000，准入失败
  副本 1：800 <= 1000，start = 15，moved = 1200，finish = 15 + ceil(1200/32) = 53
  -> 副本 1，ttft = 53；resident[1] = 800

请求 3：kv = (10+5)*4 = 60
  副本 0：420 <= 1000，start = max(16, 9) = 16，finish = 16 + ceil(60/64) = 17
  副本 1：860 <= 1000，start = max(16, 53) = 53，moved = 90，finish = 53 + ceil(90/32) = 56
  -> 副本 0，ttft = 17
```

```
placement = [0, 0, 1, 0]
ttft      = [6, 9, 53, 17]
wire      = [420, 1200]
resident  = [420, 800]
rejected  = 0
makespan  = 53
```

## 约束

- `prompt_lens[i] >= 1`、`gen_lens[i] >= 0`、`kv_bytes_per_token >= 1`、`prefill_tps >= 1`；
- `caps[d] >= 0`、`bws[d] >= 1`、`tps[d] >= 1`；
- `caps`、`bws`、`tps` 长度一致，`prompt_lens` 与 `gen_lens` 长度一致，否则就地报错；
- 请求列表可以为空，此时账本为空、`makespan` 为 0。
