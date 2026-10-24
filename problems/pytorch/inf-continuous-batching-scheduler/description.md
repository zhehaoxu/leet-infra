# LLM Serving：Continuous Batching 调度器

## 题目描述

Orca / vLLM 的**连续批处理**：不再等整批生成完，任何请求解完就释放 slot、等待队列新请求就地补位。给定 FCFS 到达序列与固定 batch 上限，模拟调度并输出每请求的**完成 step**。

## 规则

`solve(requests, max_batch_size)`：`requests` 是 $(N, 2)$ int tensor，`requests[i, 0]` 是到达 step、`requests[i, 1]` 是生成 token 数 $g_i$。总占用步 = 1 (prefill) + $g_i$ (decode)。

调度规则：
- step $t$ 从 0 起。
- 每 step：**先 admit**（按 `(arrival, id)` 升序 FCFS）等待队列里的请求进入 running 集，直至 running 数量 = `max_batch_size`。
- 每 step：**再消耗 1 步**（每 running 请求剩余 -1）。
- 剩余归 0 的请求在本 step 完成，写入 `finish[i] = t`。

返回 `finish` int tensor $(N,)$。

## 示例

```
requests = [[0,3],[0,5],[1,2],[2,4],[3,1]]，max_batch_size=2

finish = [3, 5, 6, 10, 8]
```

## 约束

- 全部 CPU；`arrival` 非降序；每请求 `gen_len >= 1`。
