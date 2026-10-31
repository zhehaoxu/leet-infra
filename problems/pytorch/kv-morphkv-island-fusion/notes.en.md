# Solution Notes · MorphKV Island Fusion

## Source basis

- Paper: MorphKV, [arXiv:2503.00979](https://arxiv.org/abs/2503.00979).
- Repository: `ghadiaravi13/MorphKV` (main, tree sha `de62fc8c…` at fetch time).
- Main logic: `LlamaAttentionMorph.morphkv_hierarchical_cache` in `morphkv/models/patch_llama.py` (lines 266-405) and its call site in `LlamaAttentionMorph.forward` (lines 477-503, the hierarchical path runs only when `'unimp' in morph_type`).
- Cache assembly: `DynamicCache.fuse_kv_hierarchical` in `morphkv/morph_cache.py` (lines 143-182, `torch.cat([fused_keys, imp_keys, window_keys], dim=2)`) and `DynamicCache.cleanup` (lines 106-133, `index_select` removal driven by the mask).

Key fragment (verbatim):

```python
past_attn = scores_per_kv[:, :, -(self.WIN_SIZE + 1):-1, :-(self.WIN_SIZE + 1)]
importance = nn.functional.softmax(past_attn / self.fuse_temperature, dim=-1).mean(dim=2)
past_val_norms = past_key_value.value_cache[self.layer_idx][:, :, :num_past, :].norm(dim=-1)
importance = importance * past_val_norms
num_important = int(self.MAX_CAPACITY * self.imp_budget)
num_fused_slots = self.MAX_CAPACITY - self.WIN_SIZE - 1 - num_important
island_ids_all = torch.cumsum(is_important, dim=-1)
weights = unimp_importance / island_mass.gather(2, unimp_island_ids).clamp(min=1e-8)
island_scores = island_mass / island_counts.clamp(min=1)
```

`imp_budget=0.5` and `fuse_temperature=1.0` come from `config.morphkv.get('imp_budget', 0.5)` (line 218) and `float(config.morphkv.get('fuse_temperature', 1.0))` (line 215); `LongBench/pred_single.py` also defaults `-ib/--imp_budget` and `-ft/--fuse_temperature` to 0.5 / 1.0.

## Upstream differences (where a literal copy would bite)

1. **Importance is also scaled by value norms.** The prose only mentions `softmax(past_attn/T).mean(heads)`, but lines 300-301 multiply `importance` by `past_val_norms = ||v||_2` over the `past` part. This problem keeps that step and lets you pass `value_norms` explicitly, so the test does not depend on the magnitude of `values`.
2. **The upstream `-1` is the current token.** `num_fused_slots = MAX_CAPACITY - WIN_SIZE - 1 - num_important`, and the `past_attn[..., -(WIN_SIZE+1):-1, ...]` slice both reserve room for the token being generated. This problem is a pure compression operator with no current token, so `n_fused = max_capacity - window_size - n_budget` and the window is just `keys[:, P:, :]`.
3. **GQA grouping.** Upstream first reshapes `scores` and reduces query groups with `.max(dim=2)` to get per-KV-head scores. Here `attn` is already per KV head, so grouping is out of scope.
4. **Upstream ties are implementation-defined.** It uses `topk(..., sorted=False)` followed by `sort()`, so which equal-score item wins is unspecified. This problem fixes the tie-break: important positions and islands both prefer the smaller index/id, then are re-sorted ascending.
5. **Empty islands are handled differently.** Upstream sets `island_counts == 0` scores to `-inf`, zero-fills the slots, and masks them in attention via `zero_pad_offset`. This problem returns no attention offsets; the same information is exposed as `island_counts == 0` / `island_idx == -1`.
6. **Accumulation precision.** Upstream accumulates with `scatter_add_` in the model dtype (bf16/fp16); this problem accumulates in fp64 and casts back, for CPU determinism.
7. **The hierarchical path is not the repo default.** `forward` only enters `morphkv_hierarchical_cache` when `'unimp' in morph_type`, while the README and `pred_single.py` default `morph_type` to `max_fused` (which goes through `morphkv_mask`'s wholesale eviction). Island fusion is thus MorphKV's "hierarchical/unimp" configuration rather than the out-of-the-box default; this problem is built on that function body.
8. Engineering details—`cleanup`'s `index_select` rebuild, the `cache_size/len` bookkeeping, and fusing only once via `fusion_done`—are all out of scope.

## Boundary with sibling problems

- `kv-snapkv-pooling`: averages window-query attention over W and smooths it with `avg_pool1d`, force-keeping the trailing window with `max+1`; it only produces scores and never rewrites K/V.
- `kv-merging-cos-nearest`: merges the lowest-score slot into a dense-region slot by **cosine/nearest neighbour**, i.e. an arbitrary pair of slots.
- `kv-wkvd-least-squares-merge`: merge weights are solved by **least squares** rather than importance weighting.
- `kv-simlayerkv`: makes a **binary keep/evict** decision per layer and performs no token-level fusion.
- This problem: top-K singletons + importance-weighted averaging **inside contiguous islands** (post-RoPE K and V fused together) + top-K islands by **mean island score**; no similarity search, no least squares, no per-layer threshold.

## Simplifications made for judgeability

- The model forward pass is removed: `attn` (window-query logits), `keys/values` (post-RoPE KV) and `max_capacity/window_size/imp_budget/fuse_temperature/value_norms` are all explicit arguments—no transformers import and no cache object.
- `attn_logit_offsets` / `zero_pad_offset` and the GQA `repeat_kv` are not returned; they are inference-framework bookkeeping unrelated to the operator's numbers.
- `important_idx / island_idx / island_counts` are returned as metadata so the layout can be verified slot by slot (otherwise a zero-filled slot is indistinguishable from a genuine zero vector).
