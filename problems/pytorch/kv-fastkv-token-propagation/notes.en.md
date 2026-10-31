# Solution Notes · FastKV Token Propagation

## In one sentence

At the TSP layer, select a token subset using scores summed across KV heads; subsequent layers carry only that subset. Their retention rate is relaxed from `retain_rate` to `retain_rate / tsp_rate`, keeping the absolute budget roughly unchanged.

## Upstream source

`dongwonjo/FastKV` (arXiv:2502.01068, seen in the README):

- `baselines/fastkv/utils.py::FastKVCluster.update_kv`: per-layer scoring and retention, with SnapKV-style window scoring (`window_size=8`, `kernel_size=7`).
- `baselines/fastkv/utils.py::compress_fastkv`: truncates by `retain_rate`; layers after the TSP layer relax the budget to `retain_rate / tsp_rate`.
- `baselines/fastkv/llama_model.py::llama_decoderlayer_forward_fastkv`: the TSP layer selects tokens, later layers reuse that subset, and `position_ids` keep their original values.
- Defaults are `tsp_idx=15`, `tsp_rate=0.2`, `retain_rate=0.1`; this problem exposes `tsp_idx` and both rates as explicit inputs.

## Boundary with existing problems

- `kv-snapkv-pooling`: shares the scoring lineage (window attention plus pooling), but this problem **does not implement pooling**; `attn_caches` supplies per-token scores directly. The focus is cross-layer TSP propagation and budget relaxation.
- `kv-simlayerkv`: makes a 0/1 lazy decision per layer (full attention vs. reuse the previous layer). This problem does not skip layers; it selects **concrete token indices** at the TSP layer and propagates them.
- `kv-key-rerotation-rope`: rewrites the RoPE phase of keys. This problem never re-rotates keys; it only ranks scores and propagates indices.

## Simplifications for judgeability

- `attn_caches` holds explicit per-layer \((T_l, H_{kv})\) scores instead of computing the upstream SnapKV scores from q/k.
- `hidden_states` and `position_ids` are explicit; `solve` only performs `index_select` and runs no model forward pass.
- `q_len` is the current segment length; `tsp_idx` is explicit; `window_size` / `retain_rate` / `tsp_rate` have defaults.
- Per-layer cache lengths are uniquely determined by the gating rule: `q_len` when `l <= tsp_idx` or `q_len <= tsp_length`, otherwise `tsp_length`.
- Ties go to the smaller index; every `kept` row is returned ascending, with the window always at the tail.
- `capacities` and per-layer `kept` are returned so budgets and selections can be checked.

## Known upstream differences

- Upstream scoring includes `kernel_size=7` pooling and window attention; here scores are inputs, so a caller using a different scoring rule will get a different ranking.
- Upstream fixes `tsp_idx=15` and takes the budget rates from configuration; this problem accepts any `tsp_idx` and rates.
- Upstream handles a decode loop and variable-length caches; this problem handles one given `q_len` segment.
- Upstream may group GQA KV heads; this problem retains each KV head independently and sums TSP scores over KV heads.
- **Same-name paper distinction**: this problem follows `dongwonjo/FastKV` (arXiv:2502.01068, training-free KV cache compression with token-selective propagation). The ACL Findings 2026 paper **"FastKV: Decoupling of Context Reduction …"** is a different work with the same name, different authors, method, and codebase; do not conflate the two or attribute the latter's claims to this problem.

## Complexity

Per-layer sorting costs \(O(\sum_l T_l H_{kv}\log T_l)\), plus \(O(T_{tsp}\log T_{tsp})\) for TSP selection. Compared with full attention's \(O(T^2)\), this is only selection and slicing overhead.
