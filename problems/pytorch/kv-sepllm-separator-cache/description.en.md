# KV Cache: SepLLM Streaming Retention of Separator Tokens

## Problem

SepLLM's (ICML 2025) observation is that **separator tokens** such as punctuation and newlines soak up far more attention than their share of the sequence. Its training-free KV compression therefore keeps only three segments—a few leading sinks, the separators seen so far in the stream, and a recent window—and throws the middle away, bounding the cache by `init_cache_size + sep_cache_size + local_size`.

The three segments are contiguous in slot space:

1. **sink prefix** `[0, init_cache_size)`, kept forever;
2. **separator region** `[init_cache_size, sep_exrange)`, holding the separator KVs seen in the stream;
3. **local window** `[L - local_size, L)`, the most recent `local_size` tokens.

One compression pass: scan the "past window" from the right edge of the separator region up to the left edge of the local window, pick the positions whose id is in `separator_token_ids`, and append them to the tail of the separator region. Once the accumulated separators exceed `sep_cache_size`, **only those with the largest positions survive**, and the right edge locks at `init_cache_size + sep_cache_size`. Upstream `compress_kv_cache_and_tokids` returns both the gathered KV and the **compacted token ids**, in the same order: sink → separators (ascending position) → local window.

## Task

Implement `solve(token_ids, sep_ids, keys, values, init_cache_size=4, sep_cache_size=32, local_size=224, boundary=0)`:

- `token_ids`: `(L,)` int64, the token id sitting in every cache slot;
- `sep_ids`: the set of separator ids, given as a list / tuple / set / int64 tensor;
- `keys`, `values`: per-layer lists of `(H, L, D)` tensors, with matching layer counts, shapes and dtype (float32/float64);
- `init_cache_size` / `sep_cache_size` / `local_size`: positive ints; `boundary`: the current right edge of the separator region, `0` meaning "not compressed yet".

Return a 5-key dict:

- `kept`: `(n,)` int64, the **original indices** of the kept slots, ascending;
- `token_ids`: `(n,)` int64, the compacted ids;
- `keys`, `values`: per layer `(H, n, D)`, gathered along the seq dimension by `kept`;
- `boundary`: int, the new right edge `init_cache_size + min(total separators, sep_cache_size)`.

Any invalid input raises `ValueError`, including upstream's `assert seq_len - local_size > sep_exrange`: when `L - local_size <= max(boundary, init_cache_size)` the past window is empty (`boundary = 0` counts as `init_cache_size` on the first compression) and upstream simply fails the assertion.

## Example

```
token_ids       = [9, 9, 1, 5, 1, 2, 3, 7, 1]   # L = 9, positions 0..8
sep_ids         = {1, 2}
init_cache_size = 2, sep_cache_size = 2, local_size = 3, boundary = 0

sink     = [0, 2)              -> 9, 9
acc      = []                  # boundary = 0: first compression, separator region empty
past_win = [2, 9-3) = [2, 6)   -> 1, 5, 1, 2     separators at 2 (id 1), 4 (id 1), 5 (id 2)
local    = [9-3, 9) = [6, 9)   -> 3, 7, 1

sep_all  = [2, 4, 5]           # 3 candidates for sep_cache_size = 2 slots
sep_keep = [4, 5]              # newest two survive, position 2 is thrown away
boundary = init_cache_size + 2 = 4

kept      = [0, 1, 4, 5, 6, 7, 8]
token_ids = [9, 9, 1, 2, 3, 7, 1]   # the 1 at position 8 rides along as local, not as a sep
gathered  = [k[:, kept, :] for k in keys]      # every layer becomes (H, 7, D)
```

Position 2 is a separator, yet it is dropped because "newest only" wins; the `1` at position 8 stays in the cache because it belongs to the local window and does not consume a separator slot.

## Constraints

- **The newest separators win**: candidates are concatenated in ascending position (accumulated region first, past window after), and when over capacity the last `sep_cache_size` are taken from the tail, which is the same as evicting the smallest positions. Keeping the oldest instead changes both `kept` and `boundary`.
- **Sinks take their slots first and never overlap the separator region**: separators inside `[0, init_cache_size)` are kept once as sinks; they neither appear twice nor count toward the separator total nor advance the boundary.
- **The first compression equals an empty accumulated region**: `boundary = 0` and `boundary = init_cache_size` must give the same result; upstream relies on `Before_First_Time_Compress_Flag` to skip one `slice(init, init)`, which would trip a `start < end` assertion.
- **The local window is never evicted**: the last `local_size` tokens are kept unconditionally, even if they are all separators.
- **The boundary is a compacted index**: `boundary` points into the new slot space of `kept` / `token_ids`, not into original positions.
- **Out of range is an error**: the past window must be non-empty (`L - local_size > max(boundary, init_cache_size)`, with `boundary = 0` counted as `init_cache_size` on the first compression), and `boundary` may only be `0` or `>= init_cache_size`.
- **No position shifting**: upstream's `enable_pos_shifting` is a model-side RoPE patch; this task only judges the cache bookkeeping.
