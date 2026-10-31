# Notes: LLM-Pruner Taylor Group-Granularity Head Importance

## Upstream source

Paper: *LLM-Pruner: On the Structural Pruning of Large Language Models* (2305.11627, NeurIPS 2023). Repo `horseee/LLM-Pruner` (fetched 2026-10-31, `main` branch, 184 py files). Line-by-line alignment against:

- `LLMPruner/pruner/hf_llama_pruner.py::TaylorImportance` (L215–344) — first-order Taylor group scoring with the attention-head branch;
- `LLMPruner/pruner/hf_llama_pruner.py::_reduce` (L221–238) — group_reduction enum (sum/mean/max/prod/first/second);
- `LLMPruner/pruner/llama_pruner.py::AttentionPrunner` (L36–63) — physical head-coupled removal (wq/wk/wv/wo shrink together, `dim/head_dim` updated);
- `LLMPruner/torch_pruning/pruner/algorithms/metapruner.py` — group framework (idx granularity and min-size alignment);
- `hf_prune.py` (L71–130) — the `--taylor` family: `vectorize`, `param_second`, `param_mix`, default `param_first`.

## Line-level semantics

1. **Saliency (hf_llama_pruner.py L256–266)**: `salience = weight * grad` elementwise. `param_second` is `w·acc_grad·w`, `param_mix` is `salience - 0.5·w·acc_grad·w` — both need a second-order gradient buffer (`acc_grad`) unavailable in a forward-only judging env, so **only `param_first` and `vectorize` are reproduced**.
2. **Attention branch (L309–327)**: o_proj reduces along out (`sum(1)`), q/k/v along in (`sum(0)`); `vectorize` is `sum(...).abs()`, the `'param' in taylor` path is `.abs().sum(...)`.
   **Upstream bug (L323)**: `elif 'param' in self.taylor == 'param':` chains as `('param' in self.taylor) and (self.taylor == 'param')` — False for `'param_first'`/'param_second'/'param_mix' → `raise NotImplementedError`. The attention branch's param path is **unreachable**; only `vectorize` works there (the q/k/v/linear branch at L277–278 has no such bug). Recorded here; this problem's `param_first` attention semantics generalize the `.abs().sum()` convention of L277–278 onto the attention layout.
3. **Group alignment (L332–337)**: when `local_norm` lengths differ across dependencies (o is d, q/k/v are head_dim), `len % min_size == 0 → view(len//min, min).sum(0)`. Summing o's length-d vector in head blocks equals `view(n_heads, head_dim).sum(dim=1)` — the origin of the **per-head scalar**: the head is the physical granularity of the attention group (AttentionPrunner removes whole heads), and block sums fold the head_dim sub-channels into it.
4. **Cross-source reduction (L340–341)**: stack then `_reduce`; upstream defaults `group_reduction="sum"` (hf_prune.py L89 passes `args.grouping_strategy`, default `sum`). `mean` differs by a constant 4 — selection unchanged.
5. **Selection**: MetaPruner eliminates ascending within groups (the `torch.argsort(imp)` semantics of `prune_local`); this problem fixes ascending stable order with smaller-index ties, and makes `num_prune` an explicit parameter (upstream derives it from `ch_sparsity`; an explicit head count is more deterministic for judging).

## Modeling decisions (deltas from the upstream framework)

- **Group framework decoupled**: upstream returns a head_dim-length vector and lets the torch_pruning dependency graph decide group granularity; this problem defines **one scalar per head** (head-aggregated block sums), mathematically equivalent to "head-granularity group + block-sum alignment" while avoiding graph tracing. Declared in both descriptions and here.
- **MHA, not GQA**: upstream LLaMA-1/2 has no GQA; q/k/v share the layout. This problem requires `d_out == d_in` and validates it.
- **No `param_second`/`param_mix`**: they require the `acc_grad` buffer, unavailable for judging.
- **float64 internally**; upstream uses fp32.

## Test design and mutations

- Hand-computed 4×4 all-positive case: o row blocks `[36, 100]`, q/k/v column blocks `[60, 76]`, imp `[216, 328]`;
- vectorize cancellation case: column block `[1,-3,1,-3]` gives 6 under param_first vs 0 under vectorize;
- **Selection-flip case** (head_dim=1): `[3,-3]` vs `[2,0]` — param_first removes head 1, vectorize removes head 0; abs timing is the core graded trap;
- Ties and extremes (num_prune=0/all), the constant-4 relation between mean and sum, random-shape comparison against an independent numpy oracle (both modes, rtol 1e-9);
- 12 validation cases: shape/dtype consistency, head_dim divisibility and positivity, mode/reduction enums, num_prune range, out≠in, 1-D weight.

Mutations: 8 tried, 7 caught (abs-after-sum / sum-after-abs-vec / o-cols-instead-rows / no-salience-mult / grad-only / view-sum-to-mean / prune-largest); `unstable-tiebreak` (stable=True → default argsort) survives because torch's CPU backend happens to be stable on ties — an **indistinguishable equivalent mutation**, recorded here. The tie-break contract remains stable=True (argsort gives no such guarantee on parallel/JIT backends).

## Dedup statement

vs 470 `sparsity-head-importance`: that one is `||W_h||_F·||X_h||_F` (no gradients, no dependency groups); this one is the `w·g` Taylor product plus the o/q/k/v four-source coupling plus dual modes. vs 582 OWL (layer-wise budgets): entirely different axis.
