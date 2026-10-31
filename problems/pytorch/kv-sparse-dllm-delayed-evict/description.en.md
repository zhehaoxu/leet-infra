# KV Cache: Sparse-dLLM Delayed One-Step Block Eviction

## Problem

A diffusion LLM (dLLM) decodes a whole block in parallel instead of one token at a time: every query in the block is known and there is no causal mask between them, so scoring must take a **bidirectional** view—all queries of the block look at the cache together. At the same time, the current block's KV still holds the intermediate state of the previous denoising round and is not final, so it cannot be a candidate.

Sparse-dLLM's ([arXiv:2508.02558](https://arxiv.org/abs/2508.02558)) `CustomCache.filter_cache` therefore **delays eviction by one step**: it scores the cache with the current block's queries while cutting the current block's KV out of the candidate set, and keeps only a `keep_ratio` fraction of the candidates. Unlike autoregressive eviction there is no sink and no recent window—the first and last tokens can be dropped just like any other.

The scoring chain is: mean over the block's queries → dot product with each candidate key → mean across heads → 1-D max pooling for smoothing → top-k.

## Rules

Implement

```
solve(k, q_block, bef_filtered_len, block_len, keep_ratio=0.5, pool_kernel_size=3)
```

Inputs:

- `k`: a float32/float64 cache of shape `(B, H_kv, L, D)` that contains the current block's KV;
- `q_block`: `(B, H_q, block_len, D)`, the current block's queries; `H_q` must be a multiple of `H_kv` (GQA);
- `bef_filtered_len`: non-negative int, how many slots precede the current block, i.e. the block occupies `[bef_filtered_len, bef_filtered_len + block_len)`;
- `block_len`: positive int, the current block length, equal to `q_block.shape[2]`; also `bef_filtered_len + block_len <= L`;
- `keep_ratio`: \((0, 1]\), the retention ratio;
- `pool_kernel_size`: positive odd int, the pooling window (upstream fixes it at 3).

Computation:

1. **Delay one step**: the candidates are `cat(k[:, :, :bef], k[:, :, bef+block_len:], dim=2)` of length \(S' = L - \text{block\_len}\) (must be \(\ge 1\)); the whole current block disappears and the two remaining pieces are stitched back in their original order.
2. **GQA**: expand the candidate KV heads with `repeat_interleave(H_q // H_kv)`; query head \(h\) uses KV head \(h // (H_q/H_{kv})\) (not a tile).
3. `avg_q = q_block.mean(dim=-2)`, `(B, H_q, D)`: the arithmetic mean of all queries in the block, bidirectional and causal-mask free.
4. `scores = avg_q @ cand.transpose(-2, -1)`, `(B, H_q, S')`.
5. `heads_avg = scores.mean(dim=1)`, `(B, S')`: average across heads first.
6. `pooled = max_pool1d(heads_avg, pool_kernel_size, stride=1, padding=pool_kernel_size // 2)`: the borders are padded with \(-\infty\), so each end only compares against the neighbours it has; `kernel=1` degenerates to the raw scores.
7. `keep_num = int(S' * keep_ratio)` (truncated, may be 0); `keep_idx = argsort(-pooled, stable=True)[:, :keep_num]`, ties going to the smaller candidate index.

Return a 5-key dict:

- `avg_q`: `(B, H_q, D)` in the input dtype;
- `scores`: `(B, H_q, S')`, the per-query-head dot products;
- `heads_avg`: `(B, S')`;
- `pooled`: `(B, S')`, the final pooled scores;
- `keep_idx`: `(B, keep_num)` int64, the kept **candidate indices** (not original cache positions).

All math is done in float64 and the four floating keys use the input dtype. Raise `ValueError` when: `k`/`q_block` is not 4-D, its dtype is not float32/float64, or the two dtypes differ; batch or `head_dim` mismatch; `H_q` is not divisible by `H_kv`; an integer argument is not an int; `block_len < 1`; `bef_filtered_len < 0`; `block_len` disagrees with the block length of `q_block`; `keep_ratio` is outside \((0,1]\); `pool_kernel_size` is not a positive odd int; the block window is out of range; or no candidate is left outside the block.

## Example

```
k        = (1, 1, 6, 2)                      # positions 2 and 3 are the current block (two 9s)
           [[[ 2,  0], [ 0,  1], [ 9,  9], [ 9,  9], [-1,  0], [ 0, -1]]]
q_block  = (1, 2, 2, 2)
           head0: [[1, 0], [1, 2]]           head1: [[0, 1], [0, 0]]
bef_filtered_len = 2, block_len = 2, keep_ratio = 0.5, pool_kernel_size = 3

cand     = positions [0, 1, 4, 5] = [[2,0], [0,1], [-1,0], [0,-1]]   # S' = 6 - 2 = 4
avg_q    = head0 [1, 1], head1 [0, 0.5]
scores   = head0 [2, 1, -1, -1]
           head1 [0, 0.5, 0, -0.5]
heads_avg = [1.0, 0.75, -0.5, -0.75]
pooled    = [1.0, 1.0, 0.75, -0.5]          # one-sided windows at both ends
keep_num  = int(4 * 0.5) = 2
keep_idx  = [0, 1]                          # original positions 0 and 1
```

The two 9s of the block never appear in `scores`; `pooled[0] = max(1.0, 0.75)` and `pooled[3] = max(-0.5, -0.75)` show that each border only compares against the neighbours it has, and zero padding is never used to lift negative scores.

## Constraints

- **Delaying one step removes the whole block**: all `block_len` slots of the current block stay out of the candidate set—it is not "drop the last query row". The remaining candidates keep their original order ("before the block" then "after the block").
- **Bidirectional block view**: `avg_q` is the arithmetic mean of the block's queries, so permuting the queries inside the block changes nothing; there is no causal mask.
- **Average across heads before pooling**: reversing the order (pool per head, then average) gives a different answer whenever the heads peak at different positions.
- **Pooling borders are one-sided**: `max_pool1d` pads with \(-\infty\); padding with 0 would lift an all-negative stretch to 0 and flatten every score.
- **The keep budget truncates**: `keep_num = int(S' * keep_ratio)` may be 0, in which case a `(B, 0)` empty index is returned instead of an error.
- **No sink / recent-window exemption**: first and last tokens are treated like any other token, and the strongest candidate may well sit at either end.
- **GQA uses `repeat_interleave`**: KV head \(h\) serves one contiguous run of query heads; tile / repeat gets the group mapping wrong.
- **`keep_idx` indexes the compacted candidates**: map back to an original cache position with `i if i < bef_filtered_len else i + block_len`.
- **Ties go to the smaller index**: upstream `topk` leaves tie order undefined; this task pins it.
- Inputs are read-only and the math is float64; no sink token is assumed and the block is not assumed to be at the end.
