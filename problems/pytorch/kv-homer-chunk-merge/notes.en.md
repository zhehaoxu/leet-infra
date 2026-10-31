# Solution Notes · HOMER Hierarchical Chunk Merging

## In one sentence

Split the context into \(2^h\) chunks as leaves, let sibling nodes average their affix and concatenate their contexts, then drop the `r` tokens with the smallest "last-row attention minus a position-dependent bias".

## Why the merge looks like this

Every chunk carries the same affix (in `make_chunks` it is literally `torch.cat([prefix_ids, context_ids[:, start:end], suffix_ids])`). Concatenating two chunks end to end would duplicate the affix \(2^h\) times and no level of the tree would shrink; averaging it keeps exactly one copy, so the length contracts to \(M_L + M_R - \text{affix}\). Upstream `Chunk.merge` fixes the order as `[prefix avg, left context, right context, suffix avg]`, and hidden states (L218-232) follow the same order as the KV cache (L268-299), with the reduction mask concatenated in the same four segments (L331-341).

## Why pruning uses +inf

Pruning takes the **smallest** entries (`topk(largest=False)`). If protection were only "set the weight to 0", a zero would be the minimum whenever every score is positive and the affix would be deleted immediately; `masked_fill(~reduction_mask, math.inf)` pushes those positions to the end (L243-245). Note that the returned tensor is the masked one (L252), so `weights` must be `+inf` on protected positions, not the raw score.

## Why the bias runs backwards

Upstream subtracts the last position from `position_ids` and multiplies by −1 to get "distance from the end" (L229-234), then indexes `bias_wrt_rel_pos[position_ids]` (L237-238). So `bias[0]` belongs to the last token and `bias[M-1]` to the first; adding the bias in forward order shifts the whole thing by one.

## Differences from upstream

- **One node only.** The full tree applies this step recursively: `_merge` first computes `l_target_len`/`r_target_len` for each subtree (L575-600; they differ when the right side is short) and then calls `__process_nonleaf` to merge and prune. A middle node's metric comes from its own forward pass, which cannot be reproduced here, so `metric` is an explicit input.
- Upstream derives `r` as `hidden_states.shape[1] - target_len` (L405) and only sets reduction info on the last layer of that node (L412-417); here `r` is a direct argument.
- Upstream's `position_ids` jump after merging (`chunk.py` L393-398 offsets the suffix to `prefix + eff_chunk_len`); this problem assumes **contiguous positions**, i.e. `rel = (M-1) - j`.
- When `r = 0` upstream short-circuits and fills `significance_weights` with placeholder ones (L300-305); here the real weights are always computed, only nothing is removed.
- Ties going to the smaller index and all shape/argument validation are additions of this problem.

## Boundaries

- `r >= M` is an error: upstream asserts `target_len > 0`, a chunk must keep at least one token.
- `r` larger than the number of removable positions is an error: protected positions sit at `+inf`, so `topk` would delete them first, which is not allowed.
- `prefix_len + suffix_len = 0` and an affix longer than either chunk are errors, mirroring upstream's `assert prefix_len > 0 or suffix_len > 0`.

## Complexity

The merge is an \(O((M_L+M_R)HD)\) copy and the prune is one \(O(M)\) topk plus one gather. The real saving is in the forward pass: each level only runs `layers_per_chunk` layers over \(2^{h}\) chunks of halved length, so total work is the sum over \(O(h)\) chunk lengths instead of the whole document.

## Source

`alinlab/HOMER` (the repository has no README and neither code nor comments mention an arXiv id, so only code paths are cited):

- `src/homer/patch_llama.py` L222-252 `_merge_power_max_last_calibrated`: L227 `metric[:, :, -1, :].max(dim=1).values`, L229-234 the rel conversion, L237-238 the bias lookup, L240 the subtraction, L243-245 `masked_fill(~mask, inf)`, L248-250 `topk(..., largest=False)`, L252 returning `(indices, significance_weights)`;
- L256-307 `_apply_pruning`: L276 `if num_tokens_to_reduce > 0`, L290-292 `prune_mask[indices_to_remove] = False`, L300-305 the `r == 0` placeholder branch;
- L389-453 `__forward_and_reduce`: L405 `r = hidden_states.shape[1] - target_len`, L412-417 reduction info only on the last layer, L442 `chunk.prune(prune_mask)`;
- `src/homer/chunk.py` L166-181 `prune` (indexes position/attention/reduction/cache with `prune_mask`), L207-354 `merge` (L218-232 the order, L268-299 the KV average, L331-341 the mask concatenation), L427 `num_full_chunks = 2 ** ceil(log2(num_chunks))`;
- `patch_llama.py` L637-645 `tree_height`, `num_chunks = 2 ** tree_height`, `layers_per_chunk = floor(num_merging_layers / (tree_height + 1))`.

## Related problems

- `kv-merging-cos-nearest`: merges evicted tokens into their cosine nearest neighbour, position-free and single-layer; this one is **position-driven tree merging** where the last-layer attention picks the victims.
- `kv-quest-page-bound`: page-level attention upper bounds; here the significance is a real attention row.
- `kv-qtrust-twohop`: query reweighting from the same batch; HOMER's metric only reads the last row and never propagates.
