# KV Cache: OTT Tracing of Low-Magnitude Tokens and the Mean Backfill

## Problem

OTT (*Accurate KV Cache Quantization with Outlier Tokens Tracing*, ACL 2025) is a modification of KIVI-style 2-bit KV quantization. KIVI keeps the newest `residual_length` tokens in fp16 because of recency; OTT additionally **refuses to let go of a small set of tokens** — not the ones with the largest attention, but the ones with the **smallest |K| magnitude**. Once such a token lands inside a 2-bit group it drags the group's min-max range, so OTT stores its full-precision K/V in a traced set and leaves only a common representative in the quantized stream.

The traced set **accumulates across quantization steps**: `sink_max_size` caps the total number of entries and `sink_num` caps how many a single step may add. Each step scores the group-aligned prefix of the fp buffer by magnitude and compares against the score of the **`sink_num`-th smallest entry already traced**: only strictly smaller candidates get in. After the merge the whole table is re-sorted by score, so old entries end up interleaved with new ones — `sink_id` is ascending by score, not append order.

The quantization kernel itself (`triton_quantize_and_pack_along_last_dim`, group-wise min-max) duplicates `quant-kivi-kv-asym` and is out of scope; what is judged here is the bookkeeping: **how much becomes quantizable, the tracing decision plus the re-sort, and the mean backfill into the quantized stream**.

## Task

Implement `solve(keys, values, quantized_len, sink_id, sink_scores, group_size=128, residual_length=128, sink_num=5, sink_max_size=32)`:

- `keys`, `values`: float32/float64 `(1, H, F, D)` of identical shape and dtype — this layer's fp buffer that is **not quantized yet** (batch is always 1, `H` is the number of KV heads);
- `quantized_len`: number of tokens already quantized, i.e. the absolute-index offset of this step's chunk;
- `sink_id`: int64 `(n,)` absolute indices already traced; `sink_scores`: `(n,)` with `keys`' dtype, their magnitude scores, ascending; `n == 0` means nothing traced yet;
- `group_size` / `residual_length` / `sink_num` / `sink_max_size`: positive ints (upstream `example.py` defaults 128 / 128 / 5 / 32).

Rules: `quant_len = max(0, (F - residual_length) // group_size * group_size)`, the chunk is the first `quant_len` tokens of the buffer; `scores[t]` is the mean of `|K|` for chunk token `t` **over all heads and all channels** (values play no part); with an empty `sink_id` take the `sink_num` smallest, otherwise the threshold is `sink_scores[sink_num - 1]`, candidates are positions with `scores <` threshold, a step adds nothing once `n >= sink_max_size`, and otherwise the `min(sink_num, candidates)` smallest candidates are taken; concatenate old and new and re-sort **stably ascending** by score; finally replace this step's newly traced tokens in the quantized stream with their per-head mean over tokens.

Return a dict of 7 keys: `quant_len` (0-dim int64), `scores` `(quant_len,)`, `new_indices` (int64, chunk-local indices in ascending score order), `sink_id` (int64, absolute indices ordered as `sink_scores`), `sink_scores`, and `keys_out` / `values_out` of shape `(1, H, quant_len, D)` with the input dtype.

## Example

```
setup: H = 2, D = 2, group_size = 2, residual_length = 1, sink_num = 2, sink_max_size = 5
       keys[0, 0, t, :] = a[t] * 1.0,  keys[0, 1, t, :] = a[t] * 3.0,  values = keys / 2
       -> scores[t] = mean(|K| over heads and channels) = 2 * a[t]

step 1: F = 5, quantized_len = 0, sink_id = [], sink_scores = []
  a         = [2.0, 0.5, 1.5, 1.0, 3.5]
  quant_len = (5 - 1) // 2 * 2 = 4                 # the 5th token stays in the residual
  scores    = [4.0, 1.0, 3.0, 2.0]                 # chunk positions 0..3
  first step, no threshold: the 2 smallest are positions [1, 3]
  new_indices = [1, 3]   sink_id = [1, 3]   sink_scores = [1.0, 2.0]
  keys_out   = [[[2.0, 2.0], [0.75, 0.75],   [1.5, 1.5],   [0.75, 0.75]],
                [[6.0, 6.0], [2.25, 2.25],   [4.5, 4.5],   [2.25, 2.25]]]
  values_out = [[[1.0, 1.0], [0.375, 0.375], [0.75, 0.75], [0.375, 0.375]],
                [[3.0, 3.0], [1.125, 1.125], [2.25, 2.25], [1.125, 1.125]]]
                 # both traced tokens share one per-head mean: (0.5 + 1.0) / 2 = 0.75

step 2: F = 8, quantized_len = 4, sink_id = [1, 3], sink_scores = [1.0, 2.0]
  a         = [2.5, 0.25, 4.0, 0.75, 3.0, 1.25, 3.5, 1.75]
  quant_len = (8 - 1) // 2 * 2 = 6                 # chunk covers absolute ids 4..9
  scores    = [5.0, 0.5, 8.0, 1.5, 6.0, 2.5]
  threshold = sink_scores[sink_num - 1] = sink_scores[1] = 2.0
  candidates (scores < 2.0) = positions [1, 3];   tracked count 2 < sink_max_size
  new_indices = [1, 3]  ->  absolute ids [1 + 4, 3 + 4] = [5, 7]  with scores [0.5, 1.5]
  merged scores [1.0, 2.0, 0.5, 1.5] -> sorted [0.5, 1.0, 1.5, 2.0]
  sink_id     = [5, 1, 7, 3]                       # old entries are interleaved, not appended
  sink_scores = [0.5, 1.0, 1.5, 2.0]
  keys_out[0, 0, :, 0] = [2.5, 0.5, 4.0, 0.5, 3.0, 1.25]   # (0.25 + 0.75) / 2 = 0.5
```

## Constraints

- **The threshold uses the configured `sink_num - 1`, not `n - 1`**: `sink_scores[sink_num - 1]` is the `sink_num`-th smallest traced score; even after 5 entries are traced the gate is still the 2nd smallest.
- **Strictly less than**: a position with `scores == threshold` does not get in.
- **The cap is tested before this step enters**: the test is `n < sink_max_size`, so one step may push the set up to `sink_max_size + sink_num - 1` entries. When the cap is already reached the step adds nothing, but `scores` and `keys_out` are still returned (with no backfill).
- **Rounding direction**: `quant_len` floors to a multiple of `group_size` and always leaves at least `residual_length` of the newest tokens; if `F < residual_length + group_size` then `quant_len == 0`, which is upstream's L361 gate failing — nothing is quantized or traced, the state is echoed back, and `sink_num` is not compared against `quant_len`.
- **The backfill averages over tokens, not over heads**: the mean is taken across *this step's newly traced* tokens, one representative vector per head, broadcast back to all of their positions; because a single vector is broadcast, gathering those indices in score order or in ascending index order gives the same result.
- **Scores ignore values**: upstream's `metric(k, v)` takes `v` in its signature and never uses it; `scores` depends on K only.
- **Raise ValueError** when: `keys`/`values` are not 4-D, not float32/float64, mismatched in shape or dtype, or batch != 1; `sink_id`/`sink_scores` are not 1-D, `sink_id` is not int64, `sink_scores` dtype differs from `keys`, or their lengths differ; `sink_id` is non-empty but shorter than `sink_num` (upstream's line would raise `IndexError`); an integer argument is not an int (`bool` counts as illegal) or `quantized_len`/`residual_length` is negative or `group_size`/`sink_num`/`sink_max_size` < 1; `sink_num > quant_len` when `quant_len > 0`.
