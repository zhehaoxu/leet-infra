# Reference: replacement semantics, the disjointness contract, and originals timing

## Implementation

1. Validation order: weight 2-D and finite → adapters non-empty, per item a dict whose **key set is exactly** name/token_indices/delta → name non-empty and globally unique → indices non-empty, per-item int (reject bool), range `[0, vocab)`, **unique within one adapter** (upstream does not check explicitly at the layer level; `index_copy` behavior with duplicate indices is undefined, so this problem pins uniqueness and raises) → delta (tensor or nested list) with row count = len(indices), width = dim, all finite.
2. **Disjointness check**: any pairwise set intersection raises. Upstream `_check_overlapping_tokens` collects index sets over `dict.fromkeys(merged_adapters + adapter_names)` — equivalent to "all adapters pairwise disjoint" (a repeated name is deduped away first). Duplicate names are rejected at the entrance here.
3. **originals first**: grab `weight[idx]` from the original weight **before** the index_copy loop — the timing of upstream `update_layer` capturing `trainable_tokens_original` pre-merge. Doing it after merging makes originals equal delta itself — this problem's second mine.
4. **index_copy in order**: start from `merged.clone()` and apply one `index_copy(dim=0, ...)` per adapter. Since disjointness is a hard contract, order does not matter (any permutation gives the same result) — a derived property, which also makes "reverse the adapter order" an **equivalent mutation**, excluded from the effective set.

## Verification

4 `check_*` functions cover: the hand example (two adapters, originals from the original rows), the replacement-vs-addition anchor (delta=[1,2] outright replaces W=[100,200]), a torch `index_copy` parity (fixed-seed random 8×5, two adapters, delta as tensor/list, a dim=1 narrow table, originals matching `W[idx]`), both overlap shapes (exact/partial), and 13 invalid inputs.

Mutation testing (4/4 effective mutations caught + 2 equivalent recorded): index_copy changed to addition (`merged[index] = merged[index] + delta`; anchor and example catch), disjointness check removed (overlap cases catch), originals moved after merge and grabbed from merged (example catches), duplicate index within one adapter allowed (validation catches). Two equivalent mutations: ① reversing the adapter application order — order-independent under the disjointness contract, a derived property; ② the `weight + weight.sum()*0` no-op, numerically identical.

## Scope and sources

Upstream huggingface/peft `src/peft/tuners/trainable_tokens/layer.py`:

- `TrainableTokensLayer.update_layer` (L109-160): `originals = self.weight[self.token_indices[adapter_name]]` (original rows captured pre-merge); with `init_weights=True`, delta starts as originals (identity at init).
- `_check_overlapping_tokens` (L162-183): `dict.fromkeys(merged_adapters + adapter_names)` collects index sets; any intersection raises `ValueError` ("Only disjunct token indices are currently supported").
- `get_merged_weights` (L220-233): per active adapter, in order, `W = W.index_copy(dim=0, index=index, source=deltas)` — the origin of replacement semantics.
- `merge` (L185-206) / `unmerge` (L208-218): merge overwrites with delta, unmerge overwrites with `trainable_tokens_original` — the point of keeping originals.
- `forward_adapters` (L235-281): `F.embedding` for Embedding, `F.linear` for Linear (LM-head weight tying); the Gemma `_get_embed_scale` wiring out of scope.

Deliberate deviations:

- **No model**: only the weight-merge semantics (embedding-weight perspective); Embedding/Linear forwards, weight tying (`tied_adapter`), and the DeepSpeed zero3 `_collect_token_weights` broadcast out of scope.
- delta comes in directly (upstream holds it as a training Parameter); the `init_weights` identity initialization out of scope.
- Uniqueness of indices within one adapter: upstream does not check it at the layer level (`index_copy` on duplicate indices is undefined); this problem pins uniqueness with an explicit error — turning undefined behavior into a deterministic contract.
- Explicit return contract: `originals` grouped by name (upstream stores a `BufferDict`).
