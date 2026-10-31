# Solution Notes · LogQuant Full-Precision Index Folding

## In one sentence

Two integer lists plus two thresholds: prefill repeatedly keeps "even positions of the first `2W` of the whole list" while `>= 3W`, decode folds the oldest `W` window entries into the same even-position subsample once `> 2W`; not a single floating-point operation.

## Why this bookkeeping works

"Log-distributed" means **the older a position is, the sparser it is sampled**: every fold thins the previous sample to its even positions, so the head doubles its spacing each round while the tail just absorbed from `local` stays dense.

- `L = 20, W = 4`: 3 folds (`20 → 16 → 12 → 8`), result `[0, 8, 12, 14]` — gaps 8 / 4 / 2, denser towards the end.
- `L = 50, W = 4`: 10 folds, result `[0, 36, 40, 42]` — the head jumps straight to 36.
- `L = 12, W = 4`: a single fold, `[0, 2, 4, 6]`.

The number of folds has a closed form: for `L >= 3W` it is `floor((L - 3W) / W) + 1`, since `len(tmp)` drops by exactly `W` per round; plugging in `L = 20, W = 4` gives 3, matching the measurement.

The decode stage is the incremental version of the same thing: `log` keeps getting thinned by powers of two at its head, while `local` holds a contiguous run of `W`~`2W` recent positions. The trigger is `> 2 * W`, not `>= 2 * W` (upstream L977), so `local` may hold the `(2W+1)`-th entry before anything fires; after dropping `W` entries it is back to `W + 1`, so two merges can never happen in one step.

## Four conventions that are easy to get wrong

- **Even positions of the concatenated list, not half of each part.** Build `tmp = log + local` first, then `tmp[0:2W:2]`; folding `log` and `local` separately loses boundary positions. In `solve(20, 0, 4)` the first round gives `log = [0,2,4,6]`, `local = [8..19]`, and the second round takes `tmp[0:8:2] = [0,4,8,10]` — `tmp[4] = 8` and `tmp[6] = 10` are borrowed from `local`.
- **`>= 3W` (prefill) and `> 2W` (decode) are not interchangeable.** One wrong sign and either `L = 3W` stops folding or `len(local) = 2W` merges too early.
- **The post-loop override keys off the original `L`**: `W < L < 3W` (upstream L969 tests `key_states.shape[-2]`, not `len(tmp)`). With `L >= 3W` no override runs even if only `2W` positions are left; `L == W` fits neither branch, so `log` stays empty.
- **The first decode merge does not thin while `log` is empty** (the `else` at upstream L980-981): `local[:W]` is moved over verbatim. Only from the second trigger on does `(log + local[:W])[0::2]` apply.

## Conventions that are ours, not upstream's

- Indices only — no quantization error and no KV rebuild; the quant kernel duplicates `quant-kivi-kv-asym` and was deliberately stripped.
- `prefill_log_index` / `prefill_local_index` are the snapshot **before the first decode step**; upstream never exports this intermediate, the problem adds it to pin the folding process.
- `prefill_len == 0` means "no prefill call at all", positions start at the first decode token. Upstream cannot produce an empty prefill (`update` always carries `key_states`); this is our convention to decouple the two stages.
- `prefill_len == 1` follows the `else` at upstream L975 and counts as one decode step, i.e. the same result as `L == 0` plus one decode step.

## What is judged

5 int64 1-D tensors, all ascending; `final_fp_positions` is the deduplicated union of `final_log_index` and `final_local_index` (the two lists are semantically disjoint, but asserting a set is safer). `window_length < 1` must raise — with `W = 0` upstream's `while len(tmp) >= 0` never terminates.

## Complexity

A naive simulation is enough: `O(L / W)` folds, each slice `O(L)`, so `O(L² / W)` overall; `L` in the thousands is still instant. Judge cases stay at `L <= 400`.

## Source

- Repository `Concyclics/LogQuantKV` (the `[Code]` link in the awesome list, seen first-hand); `src/LogQuant.py`: `LogQuantizedCache.update` L950-1046, prefill index block L961-973, decode index block L976-982, `window_length` default 64 at L853.
- README L1 links arXiv 2503.19950 directly, L21-22 state "total full precision tokens = 3*window_length" (PartialLogQuant: 2*window_length); the BibTeX entry uses key `han2025logquant`, OpenReview `E26zsSXT3J`, ICLR 2025 SLLM workshop.
- `PartialLogQuantizedCache`'s index block L1258-1281 is line-for-line identical to `LogQuantizedCache`'s; the 2W/3W difference claimed by the README only shows up outside the index block, in the per-layer content cache (L1291, L1341). This problem follows the main class.

## Related problems

`quant-kivi-kv-asym` (the quantization kernel that was stripped), `kv-quest-page-bound` (another page-level retention rule), `kv-homer-chunk-merge` (content-merging tree reduction, as opposed to this purely index-based thinning).
