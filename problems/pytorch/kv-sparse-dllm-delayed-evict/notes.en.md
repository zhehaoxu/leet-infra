# Solution Notes · Sparse-dLLM Delayed One-Step Block Eviction

## In one sentence

A dLLM decodes a whole block at once: score the cache with the mean of the block's queries while cutting the current block's KV out entirely (delayed one step), then average across heads, run one `max_pool1d(3,1,1)`, and take the top-k fraction.

## Why "delay one step"

In autoregressive decoding the new token's KV was just computed and can be scored immediately. A diffusion LLM's block is denoised over several rounds, so the current block's KV is only the **intermediate state of the previous round**; scoring it means treating noise as evidence. Upstream `CustomCache.filter_cache` therefore keeps the current block out of the candidate set for this step, letting it enter the scoring only after the block is finalized. The candidates are thus the part of the cache before the block plus the part after it, of length `L - block_len`.

Note this is not "drop only the last query row": all `block_len` slots of the block are ineligible. And because text **after** the block may exist (a dLLM sees the whole sequence), the candidates are stitched from both sides instead of being a causal prefix.

## Why the scoring chain has this order

```
cand     = cat(k[..., :bef, :], k[..., bef+block:, :], dim=-2)
cand     = cand.repeat_interleave(Hq // Hkv, dim=1)      # GQA
avg_q    = q_block.mean(dim=-2)                          # block mean, bidirectional
scores   = avg_q @ cand.transpose(-2, -1)                # (B, Hq, S')
heads_avg= scores.mean(dim=1)                            # average across heads first
pooled   = max_pool1d(heads_avg, kernel=3, stride=1, pad=1)
keep_idx = argsort(-pooled, stable=True)[:, :int(S' * keep_ratio)]
```

- **Mean over the block's queries**: all queries of the block are peers, and only their mean yields one block-level score; the last row would collapse back to the autoregressive view.
- **Average heads before pooling**: pooling smooths along **position**, so the head axis must be gone first; pooling per head and then averaging misaligns the heads' peaks and gives a different answer (the test `check_heads_averaged_before_pooling` contrasts `[0.5,0,0.5]` with `[0.75,0.5,0.75]`).
- **`max_pool1d`, not average pooling**: taking the window max drags the neighbours of a high-scoring key into the top-k together, keeping the block contiguous; `kernel=3, stride=1, pad=1` are the upstream settings.
- **One-sided borders**: PyTorch `max_pool1d` pads with \(-\infty\), so position 0 only competes with position 1. Padding with 0 the way `avg_pool1d` does would lift an all-negative stretch to 0 and flatten the scores.
- **GQA uses `repeat_interleave`**: KV head \(h\) serves a contiguous run of `Hq/Hkv` query heads; `repeat` / `tile` gets the mapping backwards.

## How this differs fundamentally from autoregressive eviction

- **View**: H2O / SnapKV / TOVA / StreamingLLM all use "causal past + the new query", so scoring covers a prefix only; here the block is bidirectional, candidates lie on both sides of the block, and every query in the block participates.
- **Timing**: autoregressive methods score the KV they just wrote; this task **delays one step** and keeps the whole current block out, so an unfinalized intermediate state is never used as evidence.
- **Protection rules**: autoregressive methods almost always exempt a sink (the first few tokens) and a recent window (the last few tokens); Sparse-dLLM's training-free setting has neither, so the first and last tokens can be evicted like any other (the test `check_block_window_is_excluded` asserts that neither `0` nor `5` survives).
- **Budget granularity**: one cut at `int(len * keep_ratio)` of the candidates, with no accumulated score and no per-layer budget.

## Differences from upstream / simplifications for judgeability

- **Granularity of the top-k**: upstream `filter_cache` runs one top-k per KV head, so each head may keep a different set; this task averages across **all** query heads into a single `(B, S')` score and runs one top-k, returning a shared `keep_idx`. That is the simplification made here for judgeability; `scores` is still per head so the upstream intermediate quantities can be checked.
- **Locating the candidates**: upstream's cache is already compacted and `filter_cache` works on the filtered sequence; this task receives the full cache `k` plus `bef_filtered_len`, which locates the block and is equivalent to upstream's "length before the block".
- **Return value**: upstream returns only the kept indices; this task also returns `avg_q` / `scores` / `heads_avg` / `pooled` so every step is checkable.
- **`keep_num = 0`**: this task allows a `(B, 0)` empty index; upstream has no explicit branch for such an extreme ratio.
- **Ties**: upstream `torch.topk` leaves tie order undefined; this task pins "smaller candidate index" via `argsort(-pooled, stable=True)`.
- Input validation and the `block_len` / `q_block` consistency check are additions of this task.

## Boundaries

- `bef_filtered_len + block_len > L`: the block window is out of range, raise.
- `L == block_len`: nothing is left outside the block, raise.
- `keep_ratio` so small that `keep_num = 0`: legal, an empty index is returned.
- `pool_kernel_size = 1`: pooling is the identity, `pooled == heads_avg`.
- `H_q == H_kv`: no replication, heads map one to one.
- Inputs are read-only; the block position is given by `bef_filtered_len`, and the block need not be at the end.

## Complexity

The per-head dot products are \(O(B H_q S' D)\), the head average and pooling \(O(B H_q S')\), and the top-k \(O(B S' \log S')\). Compared with rerunning attention over the whole cache, this step only costs an \(O(S' D)\) scoring pass.

## Source

`OpenMOSS/Sparse-dLLM` (arXiv **2508.02558**, the id is visible in the repository README):

- `opencompass/models/sparse_dllm/dream/Cache.py::CustomCache.filter_cache`: delayed one-step scoring (current block excluded), mean over the block's queries, `avg_q · K` followed by a cross-head mean, `max_pool1d(kernel=3, stride=1, pad=1)`, and a per-KV-head `topk(keep_num=int(len * keep_ratio))`;
- this task turns the same procedure into explicit inputs `k` / `q_block` / `bef_filtered_len` / `block_len`.

## Related problems

- `kv-h2o-heavy-hitters`: causal prefix + accumulated attention + sink/recent, evicting by token importance; this one is bidirectional blocks, delayed one step, with no protected slots.
- `kv-snapkv-pooling`: also smooths within a window, but with `avg_pool1d` over a **causal** window and a forcibly kept tail; this one uses `max_pool1d`, the window spans both sides of the block, and nothing is forcibly kept.
- `kv-tova-head-mean`: reads only the last query's head mean; this task takes the mean over the block's queries, a bidirectional view.
- `kv-streaming-llm-sink`: hard sink + sliding-window protection; this task explicitly has no sink / local rule, so eviction is decided by score alone.
