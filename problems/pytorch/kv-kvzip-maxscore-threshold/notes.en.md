# Solution Notes · KVzip Aggregation and the Global Threshold

## In one sentence

Take a **max** over all prefill queries for every KV pair, sort across layers and heads once, use the score at **index** `int(T*ratio)-1` as the threshold, and keep only what is strictly greater than it.

## Why a max

The three lines that matter in `_get_score`: reshape the query to `(bsz, n_kv, group, q_len, d)`, compute `softmax(qk/√d + causal)`, cut the sink prefix, then `attn.amax(dim=(-3, -2))`. A KV pair can influence the output if **any** query position or **any** query head of the group attended to it strongly, so "maximum" is the conservative estimate: a mean erases sparse spikes and a sum gets scaled by the group size and query count. Max commutes, so "group first, then queries" and a single `amax` are the same thing.

## Three off-by-ones

1. `n = max(int(T*ratio)-1, 0)`: the `-1` puts the threshold exactly on "the last element to keep";
2. `valid = score > thres` is a **strict** comparison: the threshold element itself is dropped, so at most n pairs survive — never the full ratio;
3. `max(...,0)`: when `T*ratio < 1` the index is -1, upstream clamps it to 0, the threshold becomes the maximum and the result is **empty**. That is the opposite of "always keep one"; you would have to add that guard yourself.

`keep_ratio` is `valid.float().mean()` (upstream's `r_ = 1 - rmv.mean().item()` in `prune`), and it is usually clearly below `ratio`.

## The two levels

- `pair`: stack the per-layer scores, flatten everything, one threshold shared by all layers and heads, so a strong head can eat the budget;
- `pair-uniform`: `k = int(N*ratio)` per layer and head, ranking only (no absolute threshold), and that path returns `valids, 0` directly, so the threshold is always 0;
- `--level head` is a different path (context-independent head scores) and is out of scope.

## What is judged

- `scores` must be the max over group×query (a single spike has to be captured; a mean fails);
- `threshold_index` is an index: with T=20 and ratio=0.3 it is 5, not 6;
- strictly greater: with all scores equal the threshold is that value and everything is False;
- `threshold` and `threshold_index` are 0 in both the `ratio >= 1` and the `pair-uniform` branches;
- ties go to the smaller key index (upstream's `torch.topk` leaves tie order undefined).

## Boundaries and complexity

- There is no value-norm term in upstream's scoring path (grepping `linalg.norm` finds nothing); do not add one.
- Aggregation is \(O(L H G Q N)\) and the sort is \(O(T\log T)\); both are one-off post-prefill costs, not decode hot-path work.
- Everything is compared in float64: in float32 ties near the threshold jitter with rounding, and "strictly greater" is extremely sensitive to ties.

## Source

- `attention/score.py` L44 `query_states.view(bsz, num_kv, -1, q_len, head_dim)`; L61 the softmax comment `# not fp32`; L62 `attn_weights[..., self.sink:self.sink + ctx_len]`; L63 `score = attn_weights.amax(dim=(-3, -2))  # max over group, q`.
- `attention/score.py` L88-102 `_threshold`: L94 `torch.sort(score.reshape(-1), descending=True)`, L95 `n = max(int(len(score_sort) * ratio) - 1, 0)`, L96 `thres = score_sort[n].item()`, L97 `torch.where(score > thres, True, False)`, L98-100 the `ratio >= 1` keep-all branch with `thres = 0.`.
- `attention/score.py` L104-120 `_threshold_uniform`: L111 `k = int(n_seq * ratio)`, L112 `torch.topk(score, k, dim=-1)`, L120 `return valids, 0`.
- `attention/kvcache.py` L123-139 `prune`: L126 `if "uniform" in level`, L133 `r_ = 1 - rmv.mean().item()`; L168 `lens_k_head = valid.sum(-1)  # length of retained KV per head`.
- `args.py` L4 `-r/--ratio default=0.3`, L12-16 the `--level` choices `pair/head/pair-uniform`.

## How it differs from kv-adakv-headwise

AdaKV also does a global top-k over flattened scores, but there the scores are given and the exam point is "how much to prune plus a per-head safeguard"; here the exam point is **how the score is aggregated** (max over group×query) and **the index arithmetic of the threshold**, plus a per-head `pair-uniform` branch.

## Related problems

- `kv-adakv-headwise`: cross-head flattened global top-k with a safeguard.
- `kv-chunk-independent-topk`: chunked top-k over scores that are already known.
- `kv-h2o-heavy-hitters`: evicts token by token on accumulated attention — a different granularity.
