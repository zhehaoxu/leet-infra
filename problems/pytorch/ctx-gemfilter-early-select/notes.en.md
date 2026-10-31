# Reference Solution: Interleave Expansion, the Last Query and "Sum + Zero-Padded Mean Pooling"

## Implementation

1. Validate nested shapes (equal token counts per head, equal dimensions, Q/K sharing `d`, `num_heads == kv_heads * groups`), `topk >= 1`, `kernel_size` a positive odd number, `norm > 0`, all values finite.
2. **GQA expansion** uses `repeat_interleave(groups, dim=0)` — transformers' `repeat_kv` (expand + reshape) equals interleaving along the head dim; writing `repeat` instead is the first planted trap.
3. **Last query**: `queries[:, -1:, :]` (upstream `query_states[:, :, -1:, :]`, the batch dim dropped). Changing earlier query positions must not affect the result.
4. **Inner product + head sum**: `matmul(q_last, K^T)[:, 0, :]` gives `[H, n]`, then `sum(dim=0)` collapses to `[n]`. Upstream's `sum_over_heads=True` is exactly this sum branch (`sum(dim=1, keepdim=True)`); the per-head branch is out of scope.
5. **Mean pooling**: `F.avg_pool1d(sums[None,None,:], kernel_size, padding=kernel_size//2, stride=1)`. Two defaults are exam points: padding adds **zeros** and, with `count_include_pad=True`, the **denominator stays kernel_size** — edge windows are naturally pulled down, not averaged over the actual window length.
6. **Top-k**: `k_eff = min(topk, n)` (upstream clamps before `torch.topk`, which raises on k > n); sort key `(-pooled[j], j)` — value descending, **ties by ascending index** (torch.topk's tie order is undefined; this contract pins it, and the integer test data guarantees no tie straddles the boundary — ties only occur strictly inside or outside). With `norm != 1`, `distances` are divided by `norm`.

## Correctness Checks

6 `check_*` functions cover: the worked example (hand-computed sums, pooled, both topk variants, the norm variant), 4 fixed-seed oracle cases (8-head MHA, GQA 4×3, kernel=1 identity pooling, negatives mixed with `norm=0.5`), interleave-vs-repeat discrimination (both sums hand-computed), the groups=1 MHA degenerate case hand-computed, insensitivity to non-last queries, zero-padded pooling edges (denominator constant, hand-checked), and 18 invalid inputs.

The oracle (`test_cases.py::_oracle`) is an independent pure-Python float64 rewrite (explicit interleave loop, pointwise dots, edge-clipped window sums), matching the torch float32 path within floating error; integer test data keeps orderings precision-free.

Mutation testing (6/6 real mutants caught + 1 equivalent): `repeat_interleave`→`repeat` (caught everywhere, numerics shift entirely), last query → first query, head sum → head 0 only, pooling with `count_include_pad=False`, tie-break index descending, `distances` without the norm division. Removing the `k_eff = min(topk, n)` clamp is equivalent under Python slicing (out-of-range slices truncate) and not a valid mutant — the clamp guards upstream's `torch.topk(k>n)` error path; recorded in these notes.

## Scope and Sources

Upstream SalesforceAIResearch/GemFilter (2409.17422):

- `my_baseline/GemFilter/gem_filter_utils.py::standard_dis_index` (L6-19): `torch.matmul(queries, data.transpose(-1,-2))` → `[:, :, 0, :]` → `sum_over_heads` sums `dim=1, keepdim=True` → `pool` applies `F.avg_pool1d(kernel_size, padding=kernel_size//2, stride=1)` → `torch.topk(k, dim=-1)` → `distances / norm` when `norm != 1`.
- `gem_filter_utils.py::find_context` (L22-34): scores only at `select_layer_idx` when `self.indecies is None`; `repeat_kv(key_states, num_key_value_groups)`, `query_states[:, :, -1:, :]`, `k = min(self.topk, n)`, `pool=True, sum_over_heads=True` (kernel_size left at default 5).
- `llama_select_attention.py` (transformers 4.43.3): `topk = 1024`, `select_layer_idx = 13`; select mode skips the KV cache update, and the stored indices trim inputs for all later layers.

Deliberate contract differences:

- **No model, no flash attention**: post-RoPE K, Q and the group count are explicit inputs; b=1 (upstream `assert b == 1`).
- The return contract is pinned: `indices` in "value descending, index ascending" order and the full `pooled` sequence (upstream returns only the top-k values/indices).
- The `norm` parameter is kept (upstream default 1, no division); the `sum_over_heads=False` and `pool=False` branches are not examined.
- Only scoring and selection are covered; how the indices feed later layers' KV trimming involves cache wiring and is out of scope.
