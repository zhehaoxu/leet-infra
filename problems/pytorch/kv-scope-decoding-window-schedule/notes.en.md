# Solution Notes · SCOPE's Decoding-Window Budget and Discontinuous Eviction

## In one sentence

Two integer accumulators, two gates in series, and one descending top-k: the budget grows by 1 only every `delta` tokens, `discontinuous` stacks a burst gate that fires once per `delta + 1` tokens, and the frozen prefix plus the recent tail never compete.

## Why this bookkeeping works

Each upstream `update_kv_in_decoding` call handles **one layer's** KV, so `current_decoding_step` and `jump_step` are class variables advanced "once per layer call" (`SnapKVCluster.current_decoding_step` L438; the same counters and budget formula are repeated line-for-line in `PyramidKVCluster` / `H2OKVCluster` / `ALLKVCluster` at L190 / L769 / L1019). One decoded token triggers exactly `num_hidden_layers` calls, hence

```
floor(((t * L + l) / (delta * L))) == floor(t / delta)      for l < L
```

the budget and the burst counter hold the same value across all `L` calls of a token, so this problem folds them into "advance by `L` per token". That fold is not self-proving: the grading oracle really expands the `L` layers and simulates every call, and the two implementations only agree if the identity above holds. The `l < L` bound is exactly safe — the fractional part of `t / delta` is at most `(delta - 1) / delta`, and adding `l / (delta * L) < 1 / delta` still never reaches 1.

`delta` is not a free hyper-parameter: `run_longgenbench.py` L195 comments it out and derives it from the output length instead

```
delta = (output_max_len - decoding_recent_size) // (decoding_window_size - decoding_recent_size)
```

meaning "grow the budget by 1 every `delta` tokens so that it reaches `decoding_window_size` exactly when generation ends, starting from `rec`". Outside that single line, `decoding_window_size` **never appears** in the `adaptive` / `discontinuous` branches, which is why those two branches in this problem correctly ignore it.

## Four easy mistakes

- **The counters advance per layer call, not per token.** With one increment per token, `dw` grows every token and `discontinuous`'s `evict_steps` degenerates to "evict at every step".
- **The two gates are in series**: `jump_step` only advances in calls where `k_len >= mcp + dw` (the `if` at upstream L588 comes before the `elif` at L590). Steps blocked by the budget gate neither evict nor feed `jump`, so the burst period shifts wholesale — that is what `check_gates_are_in_series` pins down.
- **`jump_step` only resets after a full `L` evicting calls** (L594-597). Since `jump_step` only ever takes the values `0, L, 2L, ...` and `delta * L` is a multiple of `L`, "crossing the threshold halfway through a token" cannot happen, which is what makes the single collapsed `jump = 0` assignment equivalent.
- **`keep = dw - rec` can be 0**: in `adaptive` the early budget *is* `rec`, so as soon as the gate passes it evicts and empties the whole candidate region, dropping the length straight to `mcp + rec`. Miss that branch and you conclude "nothing was evicted".

## This problem's own contract

- Scores come from the `attn [T, H, P+T]` argument and are indexed by **absolute position**; upstream computes `q @ kᵀ` then softmax (L527-528) and indexes by **slot number**. Softmax is monotone, so the probabilities given here need not match the real ones as long as the ordering does; columns for positions absent from the cache are simply ignored.
- Only the schedule survives out of the three upstream actions: the prefill `pooling` + clustered top-k (L474-508) duplicates `kv-snapkv-pooling` and was dropped, and the KV tensor rebuild (`gather` / `cat`) is replaced by a list-of-positions simulation.
- `init_positions` may be unordered and may hold positions above `mcp`: in the real pipeline prefill compression has already permuted the prompt cache, and what is frozen is "the first `mcp` slots".
- Ties break by earlier slot in region order — **a convention of this problem**. Upstream calls `torch.topk`, whose tie behaviour is not a published contract.
- Upstream's `__init__` L452 unconditionally asserts `decoding_window_size - decoding_recent_size > 0`, even though `adaptive` / `discontinuous` never use `decoding_window_size`. This problem keeps that constraint only for the `slide` branch and lets the others pass.
- The class counters do not reset between samples or runs (an explicit `reset()` is required). This problem models a single generation, so `cds` and `jump` both start at 0.

## Grading

4 int64 keys: `window_sizes [T]`, `cache_lens [T]`, `evict_steps [E]`, `kept_positions [H, final length]`. `kept_positions` is compared element-wise with the chosen run left in descending score order, not re-sorted by position; `evict_steps` holds **step numbers**, not slot numbers, and is an empty tensor for `"None"`.

## Complexity

`O(T * H * (P + T))` of sorting plus simulation; the judge cases stay within `T <= 400`, `H <= 8`, `P <= 512`.

## Source

- Repository `Linking-ai/SCOPE` (the organisation the awesome list's `[Code]` link points at, verified firsthand), `model/kv_utils.py`: `SnapKVCluster` L432, class counters L438-440, constructor assert L452, `update_kv` L474-508, `update_kv_in_decoding` L510-624 — `'None'` L518, `'slide'` L520-550 (gate L524, softmax-sum L529, `topk` L537), `'adaptive'` L551-581 (budget L553, `+= 1` L554, gate L556, `topk` L569), `'discontinuous'` L583-623 (budget L585, `+= 1` L586, gate L588, `jump_step <` L590, `jump_step += 1` L591, `jump_layer` L594-597, `topk` L609), `ValueError('Decoding metric not supported')` L624.
- The same schedule repeats line-for-line in `PyramidKVCluster` (L24, block L147-262) with only the class-name prefix changed, and likewise in `H2OKVCluster` (L627; budget L769, burst gate L806) and `ALLKVCluster` (L928; budget L1019, burst gate L1056) — except `H2OKVCluster`'s `'slide'` branch applies `topk` to the whole `attn_cache` (L726), and `PyramidInferCluster` (L265) has no such budget formula at all. This problem follows the `SnapKVCluster` path.
- `run_longgenbench.py`: defaults `--max_capacity_prompts 512` (L259), `--decoding_window_size 1024` (L262), `--decoding_recent_size 128` (L263); the `--delta` argument is commented out at L194 and L195 derives it instead; `decoding_window_size` / `decoding_recent_size` are written into each layer's attn config at L192-193.
- Title per README L5 and the Citation block (L187-191): *SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation*, bibkey `wu2024scope`; the arXiv Atom feed (`export.arxiv.org/api/query?id_list=2412.13649`) carries `arxiv:comment` = "ACL 2025", and the README badge cites the same 2412.13649. `discontinuous` is a branch name in the code — the README only says "Three Decoding Strategies" and gives no glossary mapping it to a paper term.
- Cross-check evidence: instantiate the real `SnapKVCluster` `L` times (one per layer, each keeping its own cache), encode the absolute position in the 4th `head_dim` channel, and share one query per token across all layers. Over 13 configurations × 4 branches, `evict_steps`, `window_sizes`, `cache_lens` and the slot-by-slot `kept_positions` all matched this problem's oracle (`ALL OK`).

## Related problems

`kv-snapkv-pooling` (the stripped prefill stage), `kv-pyramid-layer-budget` (the same schedule under PyramidKV's per-layer budget), `kv-chunk-independent-topk` (another score-ordered decoding-time cut), `kv-homer-chunk-merge` (merging instead of evicting).
