# KV Cache: Page-EntroKV's Group Rényi Entropy and Page-Level Retention

## Problem

Page-EntroKV (arXiv 2610.03135) attacks a nuisance of GQA: several query heads share one physical KV buffer, so per-query-head top-k forces the engine to keep the union of their choices — the more the group disagrees, the fatter the cache — while naive mean pooling dilutes a specialist head's preference down to `1/r`.

Its answer is to **pool the heads inside each physical group first, then run top-k at page granularity**. A sink-isolated Rényi entropy per query head stands in for "how concentrated is this head", `softmax(-entropy / tau_g)` weights the group, the `r` attention curves are merged into one, that curve is split into `block_size` pages, each page is reduced with `max`, and each head keeps its top-k pages.

The only runnable implementation upstream is `page_entrokv/core.py::PageEntroKVGroupPooler` (`forward` is 19 lines); everything else in the repo (`kernels/`, `baselines/`, `evaluation/run_guardrails.py`) is self-marked `PENDING`. This exercise is exactly the pooler's full ledger.

For every batch `b` and every physical KV head `g` (there are `H_KV`), the `r = H_Q // H_KV` query heads are grouped contiguously: `group(g) = [g*r, g*r + r)`. Three intermediates:

- **Entropy**: zero out the **first `min(sink_size, T)` positions** of the row (sink isolation), renormalize over the **remaining positions** (denominator floored at `1e-12`), then take the Rényi entropy; `alpha == 1.0` degenerates to Shannon entropy, otherwise `log(sum p^alpha + 1e-12) / (1 - alpha)`.
- **Group weights**: `softmax(-ent / tau_g)` over the `r` heads of that group only.
- **Page scores**: the pooled curve `A_pooled[g][t] = sum_i w_i * attn[b][g*r + i][t]` uses the **raw attention** (sink positions are neither zeroed nor renormalized); zero-pad it to a multiple of `block_size` and take `max` per page.

Finally each `(b, g)` keeps its top-`num_retained_blocks` pages and the token mask is the page mask expanded by `t // block_size`.

## Task

Implement `solve(attn, num_retained_blocks, num_query_heads, num_kv_heads, block_size=16, tau_g=0.5, alpha=2.0, sink_size=4)`:

- `attn`: float 3-D tensor `[B, H_Q, T]`, all three axes `>= 1`, and it **must be non-negative** ("zero padding never changes a page max" relies on this). The values themselves feed the weighted pooling and the per-page `max`, so they need not be a probability or normalized.
- `num_query_heads` must equal `H_Q`; `num_kv_heads >= 1` must divide `H_Q`.
- `block_size` is an integer `>= 1` and `sink_size` is an integer `>= 0`; `sink_size > T` is truncated to `T` (the whole row is sink).
- `tau_g > 0`, `alpha > 0`; `num_retained_blocks` is an integer with `0 <= num_retained_blocks <= nb`, where `nb = ceil(T / block_size)`.

Compute the definition above independently for each `(b, g)` and return a 5-key dict:

- `entropies`: float `[B, H_Q]`
- `group_weights`: float `[B, H_KV, r]`
- `page_scores`: float `[B, H_KV, nb]`
- `retained_pages`: int64 `[B, H_KV, nb]`, values 0/1
- `retained_tokens`: int64 `[B, H_KV, T]`, values 0/1

Raise `ValueError` when `attn` is not a float tensor / is not 3-D / contains a negative value, when any of `num_retained_blocks` / `num_query_heads` / `num_kv_heads` / `block_size` / `sink_size` is not an `int` (`bool` does not count as valid) or `num_retained_blocks < 0` / `sink_size < 0` / the other three `< 1`, when `num_query_heads != H_Q`, when `H_Q` is not divisible by `num_kv_heads`, when `num_retained_blocks > nb`, or when `tau_g` / `alpha` is not a real number or `<= 0`.

## Example

```
B = 1, H_Q = 4, H_KV = 2, T = 5, block_size = 2, tau_g = 0.5, alpha = 2.0, sink_size = 1
num_retained_blocks = 1
attn[0], rows = query heads 0..3, columns = positions 0..4
  h=0   0.10 0.20 0.05 0.40 0.30
  h=1   0.30 0.10 0.20 0.05 0.15
  h=2   0.05 0.45 0.10 0.20 0.25
  h=3   0.20 0.05 0.35 0.10 0.30

nb = ceil(5 / 2) = 3; pages cover positions {0,1} {2,3} {4, pad}
entropies       = [1.1267, 1.2040, 1.1552, 1.0454]      # over positions 1..4 only
group_weights   = [[0.5386, 0.4614], [0.4453, 0.5547]]  # kv 0 = heads 0,1; kv 1 = heads 2,3
page_scores     = [[0.1923, 0.2385, 0.2308], [0.2281, 0.2387, 0.2777]]
retained_pages  = [[0, 1, 0], [0, 0, 1]]                # kv 0 keeps page 1, kv 1 keeps page 2
retained_tokens = [[0, 0, 1, 1, 0], [0, 0, 0, 0, 1]]
```

```
tie break: B = 1, H_Q = 1, H_KV = 1, T = 4, block_size = 2, tau_g = 0.5, alpha = 2.0
num_retained_blocks = 1, sink_size = 0, attn[0][0] = [0.5, 0.5, 0.5, 0.5]

entropies       = [1.3863]        # log 4
group_weights   = [[1.0]]
page_scores     = [[0.5, 0.5]]    # tie -> lower page index wins
retained_pages  = [[1, 0]]
retained_tokens = [[1, 1, 0, 0]]
```

## Constraints

- **Entropy and pooling do not read the same row**: sink positions are zeroed and renormalized only for the entropy; pooling takes the raw `attn`. Pooling the sinks away is the usual mistake.
- **`alpha == 1.0` is its own branch**: it takes Shannon entropy; an integer `alpha` of `1` lands here too.
- **The `clamp(min=1e-12)` keeps an all-sink row finite**: when the whole row is sink, the normalized row is all zeros, yet `0**alpha = 0` and `log(0 + 1e-12)` stay finite, so no NaN appears.
- **Pad before the page max and use `ceil` for `nb`**: with `attn >= 0` the pooled values are non-negative, so the appended zeros never displace the in-page maximum; page scores are still taken over the padded page.
- **Top-k ties go to the lower page index**: upstream uses `torch.topk`, whose tie order is not a public contract; this rule is this exercise's own.
- **`num_retained_blocks` may be 0 or `nb`**: 0 gives an all-zero page mask, `nb` gives all ones.
- Grading is element-wise: the 3 float keys within `1e-5`; the 2 mask keys are int64 0/1 (not bool). `B <= 4`, `H_Q <= 16`, `H_KV <= 8`, `T <= 512`.
