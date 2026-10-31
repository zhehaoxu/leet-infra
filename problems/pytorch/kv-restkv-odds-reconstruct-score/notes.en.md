# Solution Notes · ReST-KV Odds Score and Output-Reconstruction Multiplier

## In one sentence

Recent-window attention → odds `p/(1−p)` → times "how far this value is from the representative value" → 1-D pooling → top-k: four bookkeeping stages with no trained quantity anywhere. The traps are the two **orders** (aggregate first vs score first) and the two **dead switches** (`use_norm` only in `before`, `tau` only in `after + use_wo`).

## Why this bookkeeping works

SnapKV ranks a key by the average attention the window queries give it. ReST-KV says that is not enough: a key matters when it is watched a lot (large odds) **and** its value is far from the representative value compressed out of the history (large reconstruction error). `v̄` is what the layer would output if the whole history collapsed into one token, so `‖W_o v̄ − W_o V_l‖₂` is exactly the output difference caused by dropping position `l` — that is the "layer-wise output reconstruction" of the paper title.

`before` and `after` are two expansions of the same sentence: `after` averages the `W` rows into `p̄` first and only then takes odds and the norm, `before` computes the full per-query score and averages afterwards. `p/(1−p)` is convex, so by Jensen the two score arrays are not proportional and their rankings can differ — the problem accepts both and judges them with one oracle.

## Four conventions that are easy to get wrong

- **`before` uses the full cache for the representative value**: in `O = A @ value` the `A` is the whole `(B,H,W,S)` softmax and `value` covers all `S` positions; the `[:-W]` slice happens only after the score exists. `after`'s `v̄` is weighted by the history region alone and its weights sum to `< 1` (not a convex combination).
- **`after` never reads `use_norm`, `before` never reads `tau`**: the upstream `metric_mode == "before"` branch contains no `self.tau` and the `"after"` branch no `self.use_norm`; with `use_wo=False` the `after` norm carries no power either. Those dead switches are the grading signal, not a rule we invented.
- **EMA returns only the last state**: `ema()` recurses along the query axis and then `squeeze`s it, giving `(B,H,L)` rather than the whole trajectory; `alpha` weights the **newest** item, so `alpha=1` collapses to "only the last window query" and `alpha=0` to "only the first".
- **Pooling counts the pad cells in the denominator**: `F.avg_pool1d` defaults to `count_include_pad=True`, so the first and last outputs still divide by `kernel_size`; when `L=1` the window holds one real value, `avg` shrinks it to `1/ks` while `max` (padded with `-inf`) keeps it — the two are not equivalent on degenerate lengths.

## Conventions that are ours, not upstream's

- K/V arrive **already expanded** over query heads (as at the upstream call site; `RestKVCluster` itself never calls `repeat_kv`), so `H` is always the query-head count.
- **Scores are always computed**: with `S < max_capacity_prompt` upstream returns early, but this problem still emits `attn_weights / important_score / pooled_score` and only empties the selection, so the capacity gate can be judged on its own.
- top-k is pinned to **stable descending order, ties to the smaller index**: upstream uses `topk(...).indices` and torch promises nothing about ties; `maxpool` creates exact ties in batches (neighbouring windows whose maximum comes from the same element), so a rule is mandatory.
- The mask constant is built with the attention tensor's own dtype: upstream L608 calls `torch.full(..., finfo(attn.dtype).min)` without `dtype`, so a float64 input dies with `RuntimeError: value cannot be converted to type float without overflow` — upstream can only run float32. We accept both dtypes; the observable result (masked entries have probability exactly 0) is unchanged.
- `wo` is shape-validated even when `use_wo=False`: treating it as "passed but unused" is easier to judge than allowing `None`.

## What is judged

5 keys. The three float keys keep the input dtype and no input is mutated; `selected_indices` holds history-region indices (0-based, hence also absolute) in **selection order**; `kept_positions` starts with those `k` in selection order and ends with the `W` window positions ascending. `window_size >= S`, `max_capacity_prompt <= window_size` and an even `kernel_size` must all raise — an even kernel with `padding=k//2` makes the output one cell longer than the input, which upstream silently misaligns, and we reject it outright.

## Complexity

The `before` branch builds a `(B,H,W,S,C_out)` difference tensor, dominated by `O(B·H·W·S·C_out·D)`; `after` only reaches `(B,H,L,C_out)`. Test cases stay at `S <= 11`, `W <= 6`, `D <= 3`, `C_out <= 6`, with 12 randomized oracle comparisons.

## Source

- Repository `an-yongqi/rest-kv`; `restkv/restkv_utils.py`: `RestKVCluster` L521-699 (scoring L617-656, pooling L658-680, top-k and concatenation L683-698), `ema` L287-296, the ablation twin `AblationCluster.update_kv` L313-405 (same mathematics under `ts`/`ss` names), and `init_restkv` L1229-1256 with its `window_size=32 / max_capacity_prompt=4096` fallbacks.
- README L66 and L83-89 give the paper configuration `--use_wo --use_ema --alpha 0.3 --metric_mode after --pooling adaptive --scale 2000`; L98-106 lists each default, including "tau only applies in `after` mode", which matches the code.
- arXiv 2605.08840 (2026-05-09) confirmed from the repository link on the arxiv.org abstract page; the ICLR 2026 proceedings carry the same PDF.
- Cross-check run this session on CPU: `F.avg_pool1d` / `F.max_pool1d` were spied on to capture upstream's pre- and post-smoothing score arrays — 15 seeds × {before,after} × `use_wo` × `use_norm` × `use_ema` × {avg,max}, 480 runs, **bit-identical** under `torch.equal`; selection order was then recovered from the returned cache and agrees in all 192 combinations except the tie order inside `maxpool`.
- Stripped: `pooling="adaptive"` (L662-674, hard-codes `attn_weights[0]` and derives kernel widths from sorted attention indices), `use_pyramid` (L591-602, duplicates `kv-pyramid-layer-budget`), `merge_kv` (L120-171), and the `indicator='attn'/'attn-v'/'random'` branches of `AblationCluster` (the first two are plain SnapKV, the third is random).

## Related problems

`kv-snapkv-pooling` (same recent-window attention, no odds and no reconstruction multiplier), `kv-rkv-redundancy-decay` (also combines attention with a second term before pooling and top-k), `kv-pyramid-layer-budget` (the stripped per-layer budget), `kv-ott-outlier-token-trace` (another "keep a few special tokens out of the quantized stream" route).
