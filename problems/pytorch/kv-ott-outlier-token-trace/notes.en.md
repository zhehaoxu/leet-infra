# Solution Notes · OTT Tracing of Low-Magnitude Tokens and the Mean Backfill

## In one sentence

One integer floor, one threshold comparison, one whole-table re-sort, two broadcast writes: the score is `|K|` averaged over heads and channels, and what gets traced are the **smallest** few entries, not the largest.

## Why this bookkeeping works

A 2-bit group quantizer is most afraid of "the whole group is tiny": the lower the magnitudes inside a group, the narrower `scale = (max - min) / 3`, and the larger the relative error for every other token in it. OTT parks the `sink_num` lowest-magnitude tokens in a full-precision traced set and leaves only their shared mean as a placeholder in the quantized stream — because at read time those columns get overwritten by the exact traced values anyway (`att_qkquant[:, :, :, sink_id] = att_sink`, L269-271), and the V side does the same trick (L312-314 zeroes the `sink_id` columns out of the quantized path, L323-324 adds them back through `sink_v`). So this exercise only has to settle who enters the traced set, which absolute ids they carry, and what the placeholder becomes.

The threshold is `sink_scores[sink_num - 1]` (L113), not `sink_scores[n - 1]`: once the set holds 6 entries you still compare against the **`sink_num`-th smallest** score in it, i.e. the weakest member currently enrolled. A newcomer has to be stronger (smaller magnitude) than the weakest incumbent to displace it, and after the re-sort the worst one sinks to the tail to be beaten in the next step. The cap `sink_id.shape[0] < sink_max_size` (L114) is tested **before** insertion, so a single step can push the set up to `sink_max_size + sink_num - 1` entries; upstream never trims.

Check this against step 2 of the example: `a = [2.5, 0.25, 4.0, 0.75, 3.0, 1.25, 3.5, 1.75]` with `F = 8`, `residual_length = 1`, `group_size = 2` gives `quant_len = (8 - 1) // 2 * 2 = 6`, so the last token stays in the recency segment. `scores = 2 * a[:6] = [5.0, 0.5, 8.0, 1.5, 6.0, 2.5]`, threshold `sink_scores[1] = 2.0` → candidates `0.5 / 1.5`, positions `[1, 3]`, and with the offset `quantized_len = 4` their absolute ids are `[5, 7]`. Merged with the two incumbents `[1, 3]` (scores `1.0 / 2.0`) and sorted ascending by score, `[0.5, 1.0, 1.5, 2.0]` pairs with `[5, 1, 7, 3]`: old entries are interleaved among the new ones — `sink_id` is never an append log.

## Four conventions that are easy to get wrong

- **The threshold index is the configured `sink_num - 1`**: with `sink_num = 2` and 4 traced entries the threshold is still `sink_scores[1]`; using `sink_scores[n - 1]` (the table tail, i.e. the largest score) lets nearly every candidate in.
- **Strictly less than**: `scores == threshold` does not qualify (L113 uses `<`). Entries that tie the threshold and keep entering make the tied set grow every single step.
- **The backfill averages over tokens, not over heads**: `chunk[0, :, idx, :]` is `(H, m, D)` and `mean(dim=1)` folds the `m` traced tokens, leaving one representative vector per head (L139-140). A single vector is broadcast, so gathering in score order or in ascending index order gives the same result — but do not read it as a head average, nor as one global mean over heads and channels.
- **A blocked step still returns state**: once `sink_id.shape[0] >= sink_max_size` upstream falls into the `else` (L118-119) and `sink_indices` becomes `None` → nothing is appended and nothing is backfilled, yet `scores` and `keys_out` still have to come back.

## Conventions that are ours, not upstream's

- Only the tracing bookkeeping: `triton_quantize_and_pack_along_last_dim` (group-wise min-max) is line-for-line the same as `quant-kivi-kv-asym` and is deliberately stripped; `sink_k` / `sink_v` just collect the fp16 originals for the CUDA-side attention overwrite and play no part in deciding who gets traced, so they are stripped too.
- The `max(0, ...)` around `quant_len`: upstream L104 has no guard, so `F < residual_length` yields a negative number whose slice trims from the right. The call site (L361, `F >= residual_length + group_size`) keeps that unreachable in production; here `quant_len == 0` is defined as "the step idles and echoes the state back".
- The re-sort is **stable** ascending, ties keep old entries before new ones; upstream's `sink_scores.sort()` (L134) does not pass `stable=True`, so tie order is unspecified.
- "Take the k smallest" in both the first step and the candidate step is expressed as `argsort(stable)`, ties going to the **smaller index**; upstream uses `torch.topk(..., largest=False)` (L116, L121) with the same unspecified tie order, so this is the exercise's determinism contract.
- A non-empty `sink_id` shorter than `sink_num` raises `ValueError` (upstream hits `IndexError` at L113); so does `sink_num > quant_len` when `quant_len > 0`, because upstream's `topk` would crash outright.
- The batch dimension is pinned to 1: `metric` and both backfills only ever write `k[0]`, so a larger batch would silently lose information upstream.

## What is judged

Seven keys: `quant_len` is a 0-dim int64 tensor; `scores` / `sink_scores` / `keys_out` / `values_out` keep the input dtype (float32 in, float32 out); `new_indices` are **chunk-local** indices in ascending score order while `sink_id` carries **absolute** indices sorted by `sink_scores` — two different meanings, don't mix them. Inputs must not be mutated (clone the chunk before backfilling).

## Complexity

Scores cost one pass of `O(H · quant_len · D)`, the sort `O(quant_len log quant_len)`, the backfill `O(H · m · D)`. Judged cases stay within `H, D <= 4` and `F <= 32`; the randomized cross-check adds the `quantized_len` offset and the incumbent traced length as extra dimensions.

## Source

- Repository `yisunlp/OTT` (verified through GitHub search: "code for paper Accurate KV Cache Quantization with Outlier Tokens Tracing (ACL2025 Main)", default branch `master`). In `models/utils_OTT.py`: `metric` L10-11, `quant_step` L90-166 (floor L104, threshold L113, cap test L114, `topk` L116/L121, absolute-id offset L129, whole-table re-sort L134-138, backfill L139-140), the attention-side overwrites L269-271 and L312-324, the call gate L361-362, and tracing switched off in the two shallowest layers L195-197.
- Default hyper-parameters from `example.py` L17-20: `group_size` 128, `residual_length` 128, `sink_num` 5, `sink_max_size` 32.
- README L1 carries only the title and the body has no arXiv link; the paper reference comes from `HuangOwen/Awesome-LLM-Compression`, KV Cache Compression chapter, line 2186 — arXiv 2505.10938, labelled ACL 2025, authors Yi Su, Yuechi Zhou, Quantong Qiu, Juntao Li, Qingrong Xia, Ping Li, Xinyu Duan, Zhefeng Wang, Min Zhang (read off the arXiv abstract page, v1 submitted 2025-05-16). Note that `October2001/Awesome-KV-Cache-Compression` does **not** list this paper in its Quantization chapter, so don't cite that list's numbering.
- The repo still carries its old name: `pyproject.toml` packages it as `sinkq` with the description "Sink-aware 2/4/8 bit KV Cache quantization". Irrelevant to the exercise, but a reminder that the traced set is not the same thing as an attention sink.

## Related problems

`quant-kivi-kv-asym` (the stripped quantization core, and the baseline OTT modifies), `kv-logquant-log-sparse-fold` (another "who stays in full precision" schedule, purely index-driven and magnitude-blind), `kv-capkv-leverage-score` (a different importance criterion for keeping keys).
