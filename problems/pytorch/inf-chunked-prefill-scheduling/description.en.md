# Chunked Prefill: Chunked Prefilling of Long Prompts

## Problem

The prefill of a new request consumes the compute of `prompt_len` tokens all at once; if the prompt is very long (say 8k), this step **blocks** the other requests that are decoding (causing severe jitter in inter-token latency, ITL). **Chunked Prefill** (vLLM / Sarathi-Serve) **splits** the prefill of a long prompt **into several chunks** spread over multiple scheduling steps, processing only part of it per step while letting requests already decoding keep making progress.

Each scheduling step has a **token budget** `max_num_batched_tokens` (the maximum number of tokens that can be computed in one step). The budget is first allocated to the $n_\text{dec}$ requests decoding in this step (each takes exactly 1 token), and only the remaining allowance `avail = max_num_batched_tokens - n_dec` is used to feed the next prefill chunk of this prompt. Hence the actual prefill chunk size at each step = $\min(\text{remaining prompt},\ \text{avail})$.

## Rules

Given `prompt_len` (int), `max_num_batched_tokens` (int, the per-step token budget), and `n_decode_running` (int, the number of decoding requests consuming the budget in this step):

1. Each step, decoding consumes exactly `n_decode_running` tokens of the allowance
2. The allowance available for prefill is $\text{avail} = $`max_num_batched_tokens` $-$ `n_decode_running`
3. If $\text{avail} \le 0$: the budget is entirely taken by decoding, so this prompt **cannot enter at all**, and `[]` is returned
4. Otherwise schedule repeatedly: each step processes $\min(\text{remaining},\ \text{avail})$ tokens, appending to the chunk sequence, until the remainder is 0

Return `list[int]`: the chunk sizes with which this prompt is prefilled at each scheduling step (the elements always sum to `prompt_len`; `[]` when `avail<=0` or `prompt_len==0`).

## Examples

```
prompt_len = 13, max_num_batched_tokens = 8, n_decode_running = 2
avail = 8 - 2 = 6

step1: chunk = min(13, 6) = 6   remaining 7
step2: chunk = min(7,  6) = 6   remaining 1
step3: chunk = min(1,  6) = 1   remaining 0

chunks = [6, 6, 1]   (sum = 13)
```

## Constraints

- CPU; `max_num_batched_tokens >= 1`; `prompt_len >= 0`; `n_decode_running >= 0`.
- The decoding allowance is subtracted exactly as $n_\text{dec}$ at each step (it does not change because decoding requests finish early — only the chunking of this one prompt is examined).
- Return a Python `list[int]`.
