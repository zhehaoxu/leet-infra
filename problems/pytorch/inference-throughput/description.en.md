# LLM Inference Prefill/Decode Throughput Accounting

## Problem

LLM inference has two phases: **prefill** (processes the entire prompt in parallel, compute-intensive) and **decode** (autoregressive generation one token at a time,
memory-bound). TTFT (time to first token) and per-token latency follow from them. This problem estimates the FLOPs and runtime of both phases using the $2N$ convention.

## Rules

- Implement `solve(n_params, batch, prompt_len, gen_len, effective_flops)`, returning the 4-tuple
  `(prefill_flops, decode_flops, ttft_seconds, total_seconds)`.
- **Forward FLOPs per token** uses the classic $2N$ approximation (each parameter contributes one multiply-add):
  $$ \text{flops/token} = 2 \cdot \texttt{n\_params} $$
- **prefill**: consumes `prompt_len` tokens in parallel at once (does not grow with generation length):
  $$ \text{prefill\_flops} = \texttt{batch} \cdot \texttt{prompt\_len} \cdot 2N $$
- **decode**: generates `gen_len` tokens serially, computing only 1 new token per step:
  $$ \text{decode\_flops} = \texttt{batch} \cdot \texttt{gen\_len} \cdot 2N $$
- **Runtime** uses the given effective compute throughput `effective_flops` (flop/s, already including utilization/MFU discounting):
  $$ \text{ttft} = \frac{\text{prefill\_flops}}{\text{effective\_flops}}, \qquad
    \text{total} = \frac{\text{prefill\_flops} + \text{decode\_flops}}{\text{effective\_flops}} $$
- `prefill_flops` and `decode_flops` are returned as **integers**; `ttft_seconds` and `total_seconds` are returned as **floats**.

## Examples

```
Input: n_params=7_000_000_000 (7B), batch=1, prompt_len=512, gen_len=256, effective_flops=1e14
  flops/token = 2 * 7e9 = 1.4e10
  prefill_flops = 1 * 512 * 1.4e10 = 7,168,000,000,000
  decode_flops  = 1 * 256 * 1.4e10 = 3,584,000,000,000
  ttft  = 7.168e12 / 1e14 = 0.07168 seconds
  total = (7.168e12 + 3.584e12) / 1e14 = 0.10752 seconds
Output: (7168000000000, 3584000000000, 0.07168, 0.10752)
```

## Constraints

- `n_params, batch, prompt_len, gen_len` are non-negative integers; `effective_flops > 0`
- Return the 4-tuple `(prefill_flops, decode_flops, ttft_seconds, total_seconds)`
- `total_seconds ≥ ttft_seconds`; the two time terms are checked with a relative tolerance of `1e-6`
