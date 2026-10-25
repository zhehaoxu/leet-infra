# LLM Serving: Continuous Batching Scheduler

## Problem

Orca / vLLM's **continuous batching**: instead of waiting for the whole batch to finish generating, a slot is released as soon as any request completes and a new request from the waiting queue takes its place. Given an FCFS arrival sequence and a fixed batch limit, simulate the scheduling and output the **completion step** of each request.

## Rules

`solve(requests, max_batch_size)`: `requests` is an $(N, 2)$ int tensor, where `requests[i, 0]` is the arrival step and `requests[i, 1]` is the number of generated tokens $g_i$. The total number of occupied steps = 1 (prefill) + $g_i$ (decode).

Scheduling rules:
- Step $t$ starts from 0.
- Each step: **admit first** (FCFS by ascending `(arrival, id)`) requests from the waiting queue into the running set, until the number of running requests equals `max_batch_size`.
- Each step: **then consume 1 step** (each running request's remaining count decreases by 1).
- Requests whose remaining count reaches 0 complete in this step, and `finish[i] = t` is recorded.

Return `finish`, an int tensor of shape $(N,)$.

## Examples

```
requests = [[0,3],[0,5],[1,2],[2,4],[3,1]], max_batch_size=2

finish = [3, 5, 6, 10, 8]
```

## Constraints

- Everything on CPU; `arrival` is non-decreasing; each request has `gen_len >= 1`.
