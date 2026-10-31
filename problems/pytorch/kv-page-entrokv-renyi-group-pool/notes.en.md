# Notes · Page-EntroKV's Group Pooling and Page-Level Retention

## In one line

Give every query head a sink-isolated Rényi entropy, weight the `r` heads of a group with `softmax(-entropy / tau_g)`, pool their **raw** attention into one curve, reduce each `block_size` page with `max`, and take top-k over pages.

## Why this ledger

- Under GQA one physical KV buffer is shared by `r` query heads. The README's argument: per-head top-k makes the engine keep the union of the heads' choices, up to `r` times the bloat, while mean pooling dilutes a specialist head by `(1-eps)/r`. Page granularity matches the engine's physical `[N_blocks, H_KV, B, D_head]` block layout, so a retained page maps straight onto a block-table entry.
- Entropy proxies "how concentrated is this head": the peakier the distribution, the lower the entropy, and the larger the share `softmax(-ent / tau_g)` hands it. The group is then led by one expert head instead of being averaged away by its peers.
- Sinks must be isolated: the first few positions absorb a large share of attention (attention sinks). Without removing them, every head's entropy is pushed toward a similar high value and the weights lose all discrimination. The code zeroes the first `min(sink_size, T)` positions and renormalizes over the rest.
- A page is scored by its maximum: one strongly attended token is enough to keep the page. With non-negative attention, zero-padding to a multiple of `block_size` never changes that maximum.

## Four traps

- **The entropy row and the pooling row are different rows.** `core.py` L29-31 builds the row with the sinks zeroed and renormalized, yet L32/L35 group and pool the *raw* `attn_weights`. Sharing one row variable corrupts every `page_scores` — `check_sinks_are_isolated_in_entropy_only` pins this with `[100, 100, 0.1, 0.2]`: the entropy sees only the last two positions, but page 0 still scores 100.
- **The softmax runs inside the group**: `dim=-1` over the `r` heads of `ent.view(B, H_KV, r)`, not over all heads. Groups are **contiguous** slices `h // r` (that is what `view` means), not interleaved `h % H_KV`.
- **The sign of `log(sum p^alpha + 1e-12) / (1 - alpha)`**: with `alpha > 1` both numerator and denominator are negative, so the entropy is positive; writing `alpha - 1` flips the sign. `alpha == 1.0` is its own branch taking Shannon entropy, and an integer `1` lands there too.
- **Pad before the page max, and use `ceil` for `nb`**: `pad = (block_size - T % block_size) % block_size`, and only then `view(...).max(-1)`. Doing it the other way round loses a page or turns the padding into a new one.

## This exercise's own conventions

- Upstream `forward` returns only the page mask (L44). `entropies`, `group_weights`, `page_scores`, `retained_tokens` are **this exercise's own contract**: the README's weighting formula `w = softmax(-H2_sink / tau_g)` and its block-max mapping define all four quantities, but the code does not return them; exporting them pins every step.
- Breaking ties toward the lower page index is this exercise's rule. Upstream uses `torch.topk`, whose tie order is not a public contract (equal values did come back as index 0 in a spot check, but that is an implementation detail).
- Upstream validates nothing; this exercise requires `attn >= 0` (the padding-is-max-neutral argument depends on it), `tau_g > 0`, `alpha > 0`, `0 <= num_retained_blocks <= nb`, and raises `ValueError` otherwise.
- `sink_size > T` makes the whole row sink, which is upstream's hard-coded `min(self.sink_size, T)` (L28) and is kept here; `sink_size == 0` is legal (no isolation).
- Upstream is an `nn.Module` with `@torch.inference_mode()`; this exercise is a pure function with no state and no device dependence, and it never writes to `attn`.
- An all-sink row normalizes to all zeros, yet `0**alpha = 0` and `log(0 + 1e-12)` stay finite, so the entropy is not NaN. That comes from upstream's `clamp(min=1e-12)`, not from an extra special case.

## Grading

5 keys: `entropies [B, H_Q]`, `group_weights [B, H_KV, r]` and `page_scores [B, H_KV, nb]` are float keys compared element-wise with `atol = 1e-5`; `retained_pages [B, H_KV, nb]` and `retained_tokens [B, H_KV, T]` are int64 keys compared exactly (values 0/1, not bool). Do not flatten the last two, and do not pre-reduce `page_scores` to `[B, H_KV]`.

## Complexity

`O(B * H_Q * T + B * H_KV * nb * log nb)` time, plus the five returned tensors; judged at `B <= 4`, `H_Q <= 16`, `H_KV <= 8`, `T <= 512`.

## Sources

- Awesome list `HuangOwen/Awesome-LLM-Compression`: the `### KV Cache Compression` heading is at L2010 and this entry is at L2101, reading *Page-EntroKV: Hardware-Aligned, Entropy-Weighted KV-Cache Eviction under Grouped-Query Attention*, `Arxiv 2026`, with `[Paper]` pointing at arXiv 2610.03135 and `[Code]` at `github.com/bruce12-glitch/PageEntro-KV` (seen first-hand).
- The official arXiv Atom endpoint (`export.arxiv.org/api/query?id_list=2610.03135`) returns the same title and `published` 2026-10-02; the repo's `paper/main.tex` has its `\title` at L101 with the same words.
- Implementation: `page_entrokv/core.py` (44 lines). `PageEntroKVGroupPooler` L6, `__init__` L7-15 (defaults `32/8/16/0.5/2.0/4`), `renyi_entropy` L17-22 (`eps = 1e-12` L18, the `alpha == 1.0` branch L19-20, the Rényi branch L21-22); `forward` L24-44: sink mask L27-28, isolation and renormalization L29-30, entropy L31, grouping of the raw `attn` L32, entropy grouping L33, in-group softmax L34, pooling L35, padding L36-38, page split L39, page max L40, `topk` L41, mask and return L42-44.
- The repo's own status: the README section "Pilot-tier status in this repo version" states it is a pilot tier; `page_entrokv/kernels/README.md`, `baselines/snapkv_patch.py`, `baselines/vllm_block_manager_patch.py` and `evaluation/run_guardrails.py` each contain a single line starting with `PENDING:`. This exercise follows `core.py` only and repeats no throughput or accuracy claim.
- `page_entrokv/allocator.py::largest_remainder_quotas` has `remainders` fixed at `0.0` and `adjusted` copying `quotas` verbatim, so its allocation degenerates to "give the leftover to the first layers"; it also duplicates `kv-palu-rank-budget`'s budget split, hence no separate problem.
- Cross-check evidence: the real `PageEntroKVGroupPooler` and this exercise's reference agree on the page mask in 200 random configurations (`B ∈ {1,2}`, `H_KV ∈ {1,2,3}`, group ratio `r ∈ {1,2,4,6}`, `T ∈ {5,7,13,16,17,33,40}`, `block_size ∈ {1,2,4,8,16}`, `tau_g ∈ {0.1,0.5,1,3}`, `alpha ∈ {1,1.5,2,3}`, `sink_size ∈ {0,1,4,6}`, `k` uniform in `[0, nb]`); the exported `entropies` / `group_weights` / `page_scores` match a line-by-line transcription of L26-44 within `rtol 2e-5`. Edge cases also agree: `k = 0`, `k = nb`, `T < block_size`, `T % block_size == 0`, an all-sink row, `T = 1`, group ratios 1 and 6.

## Related problems

`kv-palu-rank-budget` (same GQA grouping and per-layer budget), `kv-snapkv-pooling` (the per-head independent top-k contrast), `kv-quest-page-bound` (another page-granular selection), `kv-homer-chunk-merge` (merging instead of eviction).
