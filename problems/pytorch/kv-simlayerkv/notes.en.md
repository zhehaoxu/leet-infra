# Notes · SimLayerKV

## Intuition: layers differ in "whether", not "how much"

Cross-layer KV compression has two philosophies. PyramidKV-style (order 152) assigns each layer a different **numeric budget** (a wide-bottom pyramid). SimLayerKV (arXiv:2410.13846, `SimLayerKVPress`) observes that in most layers attention is extremely concentrated — the recent tokens plus the first few sinks swallow nearly all probability mass, so the middle cache of such a "lazy" layer is dead weight. Compression is therefore binarized: **lazy = keep head+tail only; not lazy = keep everything**. No sorting, no ratio allocation — one scalar and one comparison per layer.

## The exact test (kvpress `simlayerkv_press.py`)

```python
attn_weights = compute_window_attention(..., n_last, ...)   # attention of the last n_last queries
attn_weights = attn_weights.mean((0, 1, 2))                 # batch × heads × window all averaged → (L,)
score = attn_weights[:n_initial].sum() + attn_weights[-n_recent:].sum()
if score > lazy_threshold:   # strictly greater
    keys = cat(keys[:, :, :n_initial], keys[:, :, -n_recent + n_last:])
```

Details that are easy to get wrong:
1. **The right kept block has length `n_recent - n_last`, not `n_recent`.** The slice starts at `-(n_recent - n_last)`: attention sums over the last \(n\_recent\) keys (including the window itself), but retention stops \(n\_last\) short — the freshest window tokens refresh every step anyway.
2. **Asymmetry between test and retention**: the test sums `[-n_recent:]`, retention keeps `[-(n_recent-n_last):]`.
3. **`lazy_threshold == 1.0` is a hard switch**: the source short-circuits before anything else (its default literally means "off"). Even a delta of all mass must keep everything under θ=1.0 — the sneakiest degenerate probe here.
4. Too-short sequences (\(L\le n\_initial+n\_recent+n\_last\)) are never compressed, avoiding overlap of the head/tail blocks.

## Comparison with neighbouring problems

- `kv-pyramid-layer-budget` (152): budgets come from a **closed-form formula**, independent of attention content; here each layer's 0/1 decision is data-driven.
- `kv-duo-attention-heads` (168): binarizes along the **head** dimension (retrieval vs streaming heads); SimLayerKV binarizes along the **layer** dimension (lazy vs not), and each lazy layer's retention shape is exactly the StreamingLLM head+tail pattern (145) — think "StreamingLLM with a per-layer on/off switch".
- Unlike every top-k family (149/150/165/290), the output is a bool mask plus ratios; a lazy layer's ratio is an **integer-determined constant**, independent of how far above the threshold its score landed (the score flips the switch, it does not set the strength).

## Pitfalls

- Float summation order: the test sums non-negative numbers, so pure Python vs tensor `sum` may differ by 1 ulp; the boundary probe deliberately uses exactly representable dyadic values (0.25/0.125) to pin down \(s_l=\theta\) ⇒ not lazy.
- Fancy indexing `mask[lazy]` returns a copy; write it back.
- If no layer is lazy, `ratios` must still be \(N\) zeros (don't collapse the shape).
