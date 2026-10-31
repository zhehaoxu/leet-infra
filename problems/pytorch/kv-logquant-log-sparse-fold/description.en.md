# KV Cache: LogQuant Log-Sparse Full-Precision Index Folding

## Problem

LogQuant (arXiv 2503.19950, ICLR 2025 SLLM workshop) splits the KV cache into two piles: one 2-bit quantized, one kept **in full precision**. The full-precision pile always holds `3 * window_length` tokens (call it `W`) and is itself split in two — a **stratified sample** of positions `log_sparse_index`, plus a recent window `local_index` of `W`~`2W` positions. Once the window is full, its **oldest `W`** entries are merged into the sampled pile and the whole merged list is thinned to even positions, so the older a position is, the sparser it is sampled — hence "log-distributed".

The quantization kernel itself (quanto qint2/qint4, `q_group_size=64`, same axis as KIVI) duplicates `quant-kivi-kv-asym`, so **this problem only asks about the index schedule**: given the prefill length, the number of decode steps and `W`, which token indices end up in `log_sparse_index` / `local_index` at each stage.

Upstream `src/LogQuant.py::LogQuantizedCache.update` maintains both lists inside the `layer_idx == 0` call, and prefill and decode use different thresholds:

1. **Prefill** (one call with `L > 1` tokens): `local = range(L)`, then while `len(log + local) >= 3W` fold once — `log = (log + local)[0 : 2W : 2]`, `local = (log + local)[2W :]`, re-concatenate and test again. The loop can run several rounds, each one thinning the previous sample to its even positions. Once the loop stops, if `W < L < 3W` overwrite with `log = tmp[:W]`, `local = tmp[W:]`.
2. **Decode** (one token per call): append the position to `local`; as soon as `len(local) > 2W`, concatenate `log` with the oldest `W` entries of `local` and keep even positions (if `log` is empty those `W` entries simply become `log`, no thinning), then drop those `W` entries from `local`.

## Task

Implement `solve(prefill_len, decode_steps, window_length)`:

- `prefill_len`: non-negative int, number of tokens `L` arriving in the prefill call. Per upstream, `L == 1` falls into the decode branch (one decode step only); `L == 0` means there is no prefill call at all.
- `decode_steps`: non-negative int, how many single-token decode steps follow the prefill.
- `window_length`: positive int `W` (upstream default 64; the README examples use 4/8).

Return a dict of 5 keys, all int64 1-D tensors (positions are absolute 0-based token indices):

- `prefill_log_index` / `prefill_local_index`: the two lists at the end of the prefill stage, before any decode step;
- `final_log_index` / `final_local_index`: the two lists after all steps;
- `final_fp_positions`: sorted union of the final two lists, i.e. the positions still held in full precision.

Raise `ValueError` when `prefill_len` and `decode_steps` are both 0, either is negative, either is not an `int` (`bool` counts as illegal), or `window_length < 1`.

## Example

```
W = 4, 3W = 12, 2W = 8

solve(20, 0, 4)     # prefill only, L = 20
  local = [0..19], tmp = log + local = [0..19]                 len 20 >= 12
  fold1: log = tmp[0:8:2] = [0,2,4,6],   local = tmp[8:] = [8..19],   tmp len 16 >= 12
  fold2: log = tmp[0:8:2] = [0,4,8,10],  local = tmp[8:] = [12..19],  tmp len 12 >= 12
  fold3: log = tmp[0:8:2] = [0,8,12,14], local = tmp[8:] = [16..19],  tmp len 8  <  12
  W < L < 3W is false (L = 20), so the post-loop override does not run
  prefill_log_index   = [0, 8, 12, 14]
  prefill_local_index = [16, 17, 18, 19]
  final_log_index     = [0, 8, 12, 14]        final_local_index = [16, 17, 18, 19]
  final_fp_positions  = [0, 8, 12, 14, 16, 17, 18, 19]

solve(12, 5, 4)     # prefill L = 12, then 5 decode steps (12,13,14,15,16)
  after prefill: log = [0,2,4,6], local = [8,9,10,11]          # one fold, no override
  step 12 -> len(local)=5, step 13 -> 6, step 14 -> 7, step 15 -> 8   # 8 > 8 is false
  step 16 -> len(local)=9 > 8:
      log = ([0,2,4,6] + [8,9,10,11])[0::2] = [0,4,8,10]
      local = [8,9,10,11,12,13,14,15,16][4:] = [12,13,14,15,16]
  final_log_index   = [0, 4, 8, 10]
  final_local_index = [12, 13, 14, 15, 16]
  final_fp_positions = [0, 4, 8, 10, 12, 13, 14, 15, 16]

solve(7, 0, 4)  -> log = [0, 1, 2, 3],  local = [4, 5, 6]      # 4 < 7 < 12: override branch
solve(4, 0, 4)  -> log = [],            local = [0, 1, 2, 3]   # L == W: no fold, no override
```

## Constraints

- **A fold keeps "even positions of the first 2W of the concatenated list"**: concatenate `tmp = log + local` first, then `tmp[0:2W:2]` — i.e. the previous sample gets thinned again. Thinning `log` and `local` separately gives different positions.
- **The two thresholds differ in both size and strictness**: prefill folds while `len(tmp) >= 3W`; decode merges only when `len(local) > 2W` — the `(2W+1)`-th token triggers it, exactly `2W` does not.
- **The post-loop override keys off the original `L`**: the test is `W < L < 3W` on `L`, not on `len(tmp)` after folding; with `L >= 3W` no override happens even if only `2W` positions remain.
- **`L == W` falls between two stools**: it never enters the `>= 3W` loop nor the `W < L` override, so `log` stays empty and `local` is `range(W)`.
- **The first decode merge does not thin while `log` is empty**: the trigger moves `local[:W]` into `log` verbatim; only from the second trigger on does `(log + local[:W])[0::2]` apply.
- **At most one merge per step**: a decode step admits one token, and after dropping `W` entries `len(local)` is back to `W + 1`, so two merges can never fire in a row.
- **Indices only**: no KV rebuild and no quantization error; `final_fp_positions` is the union of the two lists, sorted ascending (each list keeps its own sampling order).
