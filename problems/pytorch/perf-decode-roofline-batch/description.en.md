# Decode Roofline Batch Ledger: KV Traffic versus the Compute Knee

## Problem

During decode every generated token forces one read of the **whole weight set** out of HBM - a cost independent of
the batch - while each sequence additionally reads its own **KV cache**, which is **strictly proportional to the
batch**. So at small batch the weight read dominates and adding sequences speeds things up almost linearly, but once
the KV traffic overtakes the weight traffic, **the per-token HBM read stops falling** and throughput slams into the
$bw/K$ wall even though the compute is nowhere near saturated. Getting this curve right tells you how large a batch
is worth building, whether the KV resident in memory pays for itself, and why long-context serving cannot reach the
headline tokens/s.

`roofline-arithmetic` computes the ceiling of a *single operator*, `inference-throughput` converts FLOPs to
throughput without any HBM term, `kv-cache-memory` only says how many bytes the cache occupies; this problem puts
them on **one time axis that varies with the batch**: bytes per step, FLOPs per step, where the knee is, and how
much batch a latency-SLO target needs.

## Rules

- Bytes read from HBM per step = **one pass over the weights** + **one KV read per sequence**; the per-token KV
  bytes follow `kv-cache-memory`'s $2\cdot L\cdot n_{kv}\cdot d\cdot T$ (the 2 is K and V), times `kv_bytes`.
- FLOPs per step follow `inference-throughput`'s $2N$ convention: $\text{flops\_step}=B\cdot 2N$.
- **The step time is the larger of the two**: $\text{step\_time}=\max(\text{bytes}/bw,\ \text{flops}/peak)$ and
  $\text{tokens\_per\_s}=B/\text{step\_time}$ (only `step_time_ms` is in milliseconds).
- **Bottleneck test** reused from `roofline-arithmetic`: arithmetic intensity at or above the ridge
  ($\text{flops}\ge\text{bytes}\cdot peak/bw$) counts as compute-bound - **equality included** - so
  `is_memory_bound` uses a strict "less than".
- **Ceiling** $\text{ceiling}=\min(peak/(2N),\ bw/K)$: the first term is the compute cap, the second is the KV-read
  cap as $B\to\infty$ (with $K=0$ only the first applies). It is also the saturation point beyond which extra batch
  buys nothing.
- **Knee** `balance_batch` is the **smallest positive integer** $B$ that is compute-bound, from
  $B\cdot(2N\cdot bw-K\cdot peak)\ge peak\cdot W$ (with $W$ the weight bytes). If the coefficient is $\le 0$ the KV
  read can never be outrun, so the answer is $-1$ and throughput merely approaches $bw/K$.
- **SLO inversion** `min_batch_for_slo`: the smallest $B$ in $[1,\texttt{max\_batch}]$ with
  $\text{tokens\_per\_s}\ge$ `target_tokens_per_s` (that function is non-decreasing in $B$, so binary search works);
  $-1$ if none qualifies, and always 1 when `target_tokens_per_s=0`.
- Input meanings, every formula and the return typing live in the `starter.py` docstring, the single source of truth.

## Examples

```
Input: n_params=6.6e9, layers=32, kv_heads=8, head_dim=128, context_tokens=2048, batch=32,
       peak_flops=312e12, bandwidth=2e12, target_tokens_per_s=2000 (7B + GQA on an A100)
  weight read = 6.6e9*2 = 13,200,000,000 bytes; KV per sequence = 2*32*8*128*2048*2 = 268,435,456 bytes
  bytes/step = 13.2e9 + 32*268,435,456 = 21,789,934,592   flops/step = 32*2*6.6e9 = 422,400,000,000
  step_time = max(21,789,934,592/2e12, 422.4e9/312e12) = 0.010894967296 s -> 10.894967296 ms
  tokens_per_s = 32/0.010894967296 = 2937.135939063217 (memory-bound: 422.4e9 < 21.79e9*156)
  ceiling = min(312e12/13.2e9, 2e12/268,435,456) = 7450.580596923828   <- pinned by KV traffic
  2N*bw - K*peak = 2.64e22 - 8.375e22 < 0 -> balance_batch = -1 (no batch can saturate the compute)
  min_batch_for_slo(2000) = 19   weight_fraction = 13.2e9/21.79e9 = 0.6057842874317886
Output: (2937.135939063217, True, 10.894967296, 7450.580596923828, -1, 19, 0.6057842874317886, 13200000000)

Input: same but context_tokens=0 (no KV, only the weight read), target_tokens_per_s=2000
  bytes/step = 13.2e9  step_time = max(13.2e9/2e12, 422.4e9/312e12) = 6.6 ms
  tokens_per_s = 32/0.0066 = 4848.484848484849 (still memory-bound)
  ceiling = 312e12/13.2e9 = 23636.363636363636   balance_batch = 312e12*13.2e9/(13.2e9*2e12) = 156
  min_batch_for_slo(2000) = 14 (each extra sequence adds 151.5 tokens/s)  weight_fraction = 1.0
Output: (4848.484848484849, True, 6.6, 23636.363636363636, 156, 14, 1.0, 13200000000)

Input: n_params=1000, layers=2, kv_heads=2, head_dim=8, context_tokens=0, batch=1,
       peak_flops=1e9, bandwidth=1e9 (built to hit the exact tie)
  bytes = 2000, flops = 2000, ridge = 1 -> flops == bytes*ridge, and a tie counts as compute-bound
  step_time = 2e-6 s -> 0.002 ms, tokens_per_s = 500000.0 == ceiling, balance_batch = 1
Output: (500000.0, False, 0.002, 500000.0, 1, 1, 1.0, 2000)
```

## Constraints

- `n_params` $\ge 1$; `layers`, `kv_heads`, `head_dim` $\ge 1$; `context_tokens` $\ge 0$; `batch` $\ge 1$;
  `peak_flops`, `bandwidth` $>0$; `weight_bytes`, `kv_bytes` $\ge 1$; `target_tokens_per_s` $\ge 0$;
  `max_batch` $\ge 1$
- Return the 8-tuple `(tokens_per_s, is_memory_bound, step_time_ms, ceiling_tokens_per_s, balance_batch,
  min_batch_for_slo, weight_fraction, flops_per_token)`: items 1, 3, 4, 7 are floats (`atol=1e-5, rtol=1e-5`),
  item 2 is a bool, items 5, 6, 8 are exact ints
- Only HBM reads (weights + KV) and dense compute are modelled: no prefill, no scheduler overhead, no check that
  the memory can actually hold that batch; `balance_batch` is a physical knee and is not clipped by `max_batch`
