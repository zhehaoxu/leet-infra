# Solution Notes · SepLLM Separator Cache

## In one sentence

One compression step selects and gathers the three segments "sink prefix + separator region + local window": separator candidates come in ascending position, over capacity only the newest `sep_cache_size` survive, and the new right edge is `init + min(#separators, s)`.

## Why this bookkeeping works

The three segments are contiguous in slot space and, by precondition, never overlap, so compression degrades to pure index arithmetic — upstream `compress_kv_cache_and_tokids` likewise concatenates `[initial_kv, sep_kv, local_kv]` and compacts the token ids along the way.

```
right   = init if boundary == 0 else boundary      # first compression: accumulated region is empty
sep_all = list(range(init, right))                 # separators accumulated so far
sep_all += [p for p in range(right, L - local) if ids[p] in sep_set]   # scan the past window
sep_all = sep_all[-cap:]                           # newest separators win
kept    = list(range(init)) + sep_all + list(range(L - local, L))
new_boundary = init + len(sep_all)
```

`sep_all` is ascending by construction: the accumulated region comes first and the past window after it, so taking the tail slice is exactly "evict the smallest positions"; concatenating the three segments gives the final layout, the `index_select`-gathered KV stays in the same order as the compacted `token_ids`, and the next round's `boundary` therefore indexes directly into the compacted space.

## Three conventions that are easy to get wrong

1. **Over capacity keeps the newest, not the oldest**: the upstream `USE_MAX_SEP_CACHE` branch slices `(sep_len - max_sep_len, sep_len)` — the tail (wheel L917-925); keeping the oldest instead changes both `kept` and `boundary`.
2. **`boundary = 0` only stands for an empty separator region on the first compression**: upstream L886-887 folds `sep_exrange <= 0` into `init`, then L889 asserts the past window is non-empty; the `Before_First_Time_Compress_Flag` at L894 skips the accumulated-region slice at L897, because `slice(init, init)` would trip the `start < end` assertion at L1227.
3. **Separators inside the sinks or the local window consume no quota**: those in `[0, init)` are kept once as sinks and never counted; the last `local_size` slots are kept unconditionally; only the past window `[right, L - local)` is scanned for candidates.

## What is judged

- `kept`: int64, strictly ascending, total length ≤ `init + s + local`, with exactly the sinks in front and exactly the local window at the tail;
- `token_ids` and the per-layer `(H, n, D)` KV are gathered along `kept` in the same order with the dtype preserved, checked against a numpy oracle with tolerances 1e-5 (float32) / 2e-9 (float64); `boundary` is a Python `int` equal to `init + min(total separators, s)`, pointing into the compacted slot space;
- inputs are read-only; the gather happens in float64 and returns to the input dtype; a non-contiguous view and a contiguous copy give the same result; invalid shape/dtype, non-int parameters, `boundary ∈ (0, init)` and an empty past window all raise `ValueError` — no silent truncation.

## Boundaries

- **An empty past window is an error**: `L - local_size <= max(boundary, init)` (with `boundary = 0` counted as `init`) is precisely upstream's assertion at L889, turned into a `ValueError` here, which additionally keeps the three segments disjoint.
- **Boundaries where nothing is added or consumed**: with no new separators in the window and the region under capacity, `boundary` stays put and the accumulated region survives intact (the quiet scenario in `check_region_grows_under_cap`); a local window full of separators is still kept wholesale, consuming no `sep_cache_size` quota and advancing no boundary (the tail scenario in `check_sink_and_local_do_not_consume_budget`).
- **A streamed multi-round check**: `check_chained_stream` compresses 4 times, appending 4 tokens each step; on a long enough stream the edge finally locks at `init + s` and the cache never exceeds `a + s + w`.

## Complexity

Scanning the past window is \(O(L)\) and the gather is \(O(\sum_{\text{layers}} H\cdot n\cdot D)\) with \(n\le a+s+w\), independent of the stream length; the only extra memory per step is the compacted KV — the training-free configuration `a=4/s=128/w=256` sums to 388, still leaving room under `cache_size=512` (README L534 requires `a+s+w<c`).

## Source

`HKUDS/SepLLM` (ICML 2025, arXiv 2412.12094). The training-free implementation ships inside `package/transformers-4.38.0.post1+sepllm-py3-none-any.whl`; every citation below was checked line by line against the extracted `transformers/cache_utils.py`:

- L337 `class SepCache`; L640-641 `sep_exrange = 0` and `max_sep_exidx = init + sep_cache_size`; L679 the llama3 punctuation id table;
- `compress_kv_cache_and_tokids_layer_wise` L857-940: L886-887 the first-compression fold, L889 the assertion, L894 the first-time flag, L897/899/902 the three slices `[init, sep_exrange)`, `[sep_exrange, L-w)`, `[L-w, L)`, L917-925 the over-capacity tail slice and locked right edge, L936 the `[initial_kv, sep_kv, local_kv]` concatenation;
- `compress_past_win_2_seps` L943-992: L945-948 OR-ing `separator_token_ids` into a boolean mask and keeping positions in ascending order; `update_kv_cache_and_past_tok_ids` L817 defaults `SEP_ACCUMULATION=True`, which combined with `USE_MAX_SEP_CACHE=True` yields this problem's convention;
- `slice_kv_cache_and_tokids` L1138 with L1227 `assert start < end`; the same-named routine in the streaming branch `Streaming-SepLLM/sepllm_kv_cache/kv_cache_manager.py::compress_kv_cache_and_tokids` L257-321 is structurally identical; the default parameters `a=4/s=128/w=256/c=512/USE_MAX_SEP_CACHE=True` come from `TrainingFree-SepLLM/demo.py` L27;
- the position shift belongs to the model side: `apply_shifted_pos_emb` L995 in the wheel and `enable_pos_shifting.py::enable_pos_shifting` L4 in the streaming branch — not modelled here (README L53: training-free settings need no PE shifting).

**Additions of this problem**: a single sequence with no batching (upstream pads/truncates inside a batch via `SEP_PADDING_IN_BATCH`); no `left_padding_offset`; `boundary` and the separator id set are explicit arguments instead of per-layer internal state; the output returns the `kept` indices explicitly and gathers with `index_select`, doing no slot reuse and no numerical merging of KV; upstream asserts are converted into `ValueError`s plus added shape/dtype/parameter validation, and `boundary` is pinned to a Python `int`.

## Related problems

- `kv-streaming-llm-sink`: likewise keeps "sink + recent window" but throws the middle away wholesale; this problem adds a **content-selected** separator region with its own capacity and newest-first eviction.
- `kv-chunkkv-cumulative`: keeps whole semantic chunks with chunk-level scoring; separators here are **scattered points**, not contiguous chunks, and eviction looks only at position.
- `kv-h2o-heavy-hitters`: evicts by accumulated attention; this problem never reads attention, only token ids and positions.
