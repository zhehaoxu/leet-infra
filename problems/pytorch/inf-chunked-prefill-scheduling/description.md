# Chunked Prefill：长 prompt 的分块预填充

## 题目描述

一条新请求的 prefill 要一次性吃掉 `prompt_len` 个 token 的算力；如果 prompt 很长（比如 8k），这一步会把正在 decode 的其他请求**堵在后面**（首 token 延迟 ITL 剧烈抖动）。**Chunked Prefill**（vLLM / Sarathi-Serve）把长 prompt 的 prefill **切成若干块**，分散到多个调度步里，每步只处理一部分，同时让已在 decode 的请求继续推进。

每个调度步有一个 **token 预算** `max_num_batched_tokens`（一步最多能算多少 token）。预算先分配给本步在 decode 的 $n_\text{dec}$ 个请求（每个恰好 1 个 token），剩下的额度 `avail = max_num_batched_tokens - n_dec` 才拿来喂给这条 prompt 的下一个 prefill 块。于是每步实际 prefill 的 chunk 大小 = $\min(\text{剩余 prompt},\ \text{avail})$。

## 规则

给定 `prompt_len`（int）、`max_num_batched_tokens`（int，每步 token 预算）、`n_decode_running`（int，本步占用预算的 decode 请求数）：

1. 每步 decode 固定消耗 `n_decode_running` 个 token 额度
2. 可用于 prefill 的额度 $\text{avail} = $`max_num_batched_tokens` $-$ `n_decode_running`
3. 若 $\text{avail} \le 0$：预算已被 decode 占满，本 prompt **一步都进不来**，返回 `[]`
4. 否则反复调度：每步处理 $\min(\text{剩余},\ \text{avail})$ 个 token，追加到 chunk 序列，直到剩余为 0

返回 `list[int]`：该 prompt 在各调度步被 prefill 的 chunk 大小（元素和恒等于 `prompt_len`；`avail<=0` 或 `prompt_len==0` 时为 `[]`）。

## 示例

```
prompt_len = 13, max_num_batched_tokens = 8, n_decode_running = 2
avail = 8 - 2 = 6

step1: chunk = min(13, 6) = 6   剩余 7
step2: chunk = min(7,  6) = 6   剩余 1
step3: chunk = min(1,  6) = 1   剩余 0

chunks = [6, 6, 1]   (和 = 13)
```

## 约束

- CPU；`max_num_batched_tokens >= 1`；`prompt_len >= 0`；`n_decode_running >= 0`。
- 每步 decode 额度按 $n_\text{dec}$ 原样扣除（不因为 decode 请求提前结束而变化——只考察这一条 prompt 的切块）。
- 返回 Python `list[int]`。
