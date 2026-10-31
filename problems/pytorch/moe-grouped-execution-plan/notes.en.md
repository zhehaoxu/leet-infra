# Reference Approach: Sort the Slots into Groups, Then Restore and Merge

## Implementation

1. `flat_ids = expert_ids.reshape(-1)`; `perm` is the second output of `torch.sort(flat_ids, stable=True)` and `sorted_ids` is the first.
2. `token_rows = torch.div(perm, k, rounding_mode="floor")`; `hidden[token_rows]` is then exactly the input matrix a grouped kernel would consume.
3. `gather_ids = torch.clamp(sorted_ids, max=E - 1)`, `grouped = hidden[token_rows] * diag[gather_ids]`, then `grouped.masked_fill(sentinel.unsqueeze(-1), 0.0)` to wipe whole sentinel rows. The order matters: clamping keeps the gather in range, the post-mask is what removes every trace of the sentinel row.
4. `counts = torch.bincount(flat_ids[flat_ids < E], minlength=E)[:E]`, `offsets = torch.cumsum(counts, 0)`.
5. `inv_perm = torch.empty(S, dtype=torch.int64)`; `inv_perm[perm] = torch.arange(S)`. This is the only correct way to scatter an inverse permutation — reversing the assignment just reproduces `perm`.
6. `out = (grouped * flat_weights[perm].unsqueeze(-1))[inv_perm].reshape(T, k, D).sum(1)`.

Four common mistakes:

- **Forgetting the stable sort**. Row order inside one expert is irrelevant to a real kernel, but the concrete contents of `perm`, `inv_perm` and `token_rows` are fully determined by it, and the judge compares those tables element by element.
- **Writing `token_rows = perm % k` instead of `perm // k`**: every shape still lines up, only the numbers are wrong, which is easy to miss.
- **Not zeroing sentinels**. After the clamp, a sentinel row picks up expert $E-1$'s real output and gets multiplied by its own weight, so a dropped token silently enters the result.
- **Scattering in the `index_add_` style**. The problem fixes the accumulation order of `out` to $j=0,\dots,k-1$, and only unsort-then-reshape-sum gives it.

## Correctness checks

The 7 `check_*` functions cover: the hand example (all seven returned items checked value by value), randomized differential cases (8 shape/sentinel-count combinations plus two noncontiguous layouts), plan invariants (`perm` is a permutation, `inv_perm[perm] == arange`, `token_rows == perm // k`, sorted expert ids non-decreasing, the tail is exactly the sentinels, `counts.sum() == offsets[-1] == valid`, each grouped row equals `hidden[token] * diag[expert]`, and `out` agrees with a direct per-token accumulation that never sorts), all-sentinel and zero-weight batches, single paths (`k=1`, one token with one expert, every slot on the same expert), invalid inputs, and the maximum shape `T=k=8, E=8, D=4`.

The oracle is pure Python: a stable sort via `sorted(range(S), key=lambda s: (ids[s], s))`, grouped rows built with comprehensions, counts/offsets by a hand-written prefix sum, `inv_perm[s]` filled in a loop, and each token accumulated along $j$ with `math.fsum`. It uses no torch `sort`/`bincount`/`cumsum`/`gather` and no broadcasting.

Mutation testing: `token_rows` by modulo, no clamp, no sentinel zeroing, counting sentinels into the last expert, an inverted `inv_perm`, no unsort at all, and an unsort written as a wrong `index_copy_` mapping — all seven are caught. A `torch.sort` without `stable=True` happens to agree with the stable order on small inputs on this machine and only shows up in the 64-slot maximum-shape case, which makes that case the necessary pin for the tie convention rather than filler.

## Scope and sources

The slot algebra comes from `transformers/integrations/moe.py::grouped_mm_experts_forward` in the transformers 5.x source available to this repository's judge. Line-by-line correspondence (gate/up double projection and the activation omitted):

```text
expert_ids_g, perm = torch.sort(expert_ids.reshape(-1))
selected_hidden_states_g = hidden_states[perm // num_top_k]
sample_weights_g = sample_weights[perm]
histc_input = expert_ids_g.float() if device.type in ("cpu", "mps") else expert_ids_g.int()
tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
sentinel_mask = (expert_ids_g >= self.num_experts).unsqueeze(-1)
expert_ids_g.clamp_(max=self.num_experts - 1)
proj_out = proj_out.masked_fill(sentinel_mask, 0.0)
weighted_out = proj_out * sample_weights_g.unsqueeze(-1)
inv_perm = torch.empty_like(perm)
inv_perm[perm] = torch.arange(perm.size(0), device=device)
weighted_out = weighted_out[inv_perm]
final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
```

Upstream's own comment on sentinels matches the statement here: leave `expert_ids` unclamped so the sort pushes sentinels to the tail, `histc(max=num_experts - 1)` drops them from `tokens_per_expert`, and grouped_mm skips rows beyond `offsets[-1]`; the kernel leaves those tail rows uninitialized and one pre-mask plus one post-mask covers the forward pass. This problem turns "uninitialized" into "zeroed", because on CPU there is no uninitialized memory to lean on.

**Deliberate differences from upstream**: `counts` uses `bincount` rather than `histc` (upstream's dtype branch exists because `histc` takes floats on CPU and ints on CUDA under deterministic algorithms, which is irrelevant here); `offsets` is returned as int64 instead of int32; the sort is explicitly stable; and the expert body is replaced by a single diagonal product. The `batched_mm` path (replicating each token `num_top_k` times and using a batched matmul) is out of scope, capacity-style dispatch lives in `moe-capacity-dispatch`, and all-to-all dispatch and combine in `moe-all-to-all-dispatch`.
