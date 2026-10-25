# StreamingLLM: Sink + Sliding-Window KV Scoring

## Problem

StreamingLLM (Xiao et al., ICLR 2024) observes that LLMs dump a large amount of attention on the first few tokens—the so-called **attention sink**. The KV cache therefore only needs to keep "**sink + most recent window**", discarding the entire middle section.

`StreamingLLMPress` in kvpress expresses this policy with a **binary scoring vector**: all positions are initialized to 1, the **middle segment** to be pruned is set to 0 as one contiguous run, and `ScorerPress`'s `topk(n_kept)` then selects the positions to keep.

Given a cache length `k_len`, a sink count `n_sink`, and a compression ratio `compression_ratio ∈ [0, 1)`, first compute `n_pruned = k_len - int(k_len * (1 - compression_ratio))`, then return a float score vector of length `k_len`: positions `[n_sink, n_sink + n_pruned)` are set to **0** and all other positions to **1**.

## Rules

Implement `solve(k_len, n_sink, compression_ratio)` → return a float32 tensor of shape `(k_len,)`.

## Examples

```
k_len = 10, n_sink = 2, compression_ratio = 0.4
n_pruned = 10 - int(10 * 0.6) = 10 - 6 = 4
scores = [1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
        └ sink ┘ └ pruned ┘ └ window ┘
```

## Constraints

- CPU float32; `int(k_len * (1 - cr))` uses Python/torch truncation toward zero (not round).
- `n_sink + n_pruned <= k_len` (guaranteed by the problem inputs).
