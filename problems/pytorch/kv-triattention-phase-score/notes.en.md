# Solution Notes · TriAttention Phase Scoring

## In one sentence

After inverting RoPE, write the complex product of `q_mean` and `k_pre` as "magnitude × cos(phase)", add a temporal term built from geometric offsets to the phase, z-normalize per `(l,h)` along tokens, average across layers, and top-k per head.

## Upstream source

`WeianMao/triattention`:

- `longlive/kv_compression.py::_compute_layer_head_scores`: the main path.
  - inverse RoPE: `k_pre = k_post * exp(-iθ)`;
  - `relative = q_mean * conj(k_pre)`, whose phase is \(\varphi\);
  - `phase = Δt·ω·temporal_mask + φ`;
  - \(\Delta t\) uses geometric offsets `1, 2, 4, …`;
  - `score = Σ_f amp·cos(phase) + Σ_f extra`;
  - z-normalize per `(l,h)` along tokens, then average across layers;
  - budget constants `KV_BUDGET=2048`, `DIVIDE_LENGTH=128`, `WINDOW_SIZE=128`;
  - explicitly `raise`s for GQA.
- arXiv: the README badge cites `2604.04921`; only the badge was seen in this pass, not the paper body, so the problem statement cites it as a "README badge".

## Boundary with existing problems

- `kv-key-rerotation-rope`: re-rotates KV keys into a shared RoPE frame and **rewrites the keys**. This problem only borrows "invert RoPE" to compute a phase score; it **does not re-rotate or write keys back**, and returns scores plus token indices.
- `kv-expected-attention`: estimates expected attention from q/k statistics and selects tokens from it. This problem is a complex phase/amplitude geometric score followed by z-normalization and layer averaging; the scoring formula differs.
- `kv-snapkv-pooling`: per-key scores from window attention plus 1D pooling. Here the score sums phase terms over frequencies and involves no pooling.

## Simplifications for judgeability

- `q_mean` / `q_abs` / `k` / `rope_cos` / `rope_sin` / `omega` / `pos` are all explicit inputs; upstream reads them from model hidden states and the RoPE cache.
- `keep` is an explicit budget replacing the `KV_BUDGET` / `WINDOW_SIZE` eviction flow; `temporal_count` and `offset_max_frames` are explicit as well.
- `normalize_scores` can be turned off to inspect the raw `score`; `score_aggregation` switches between `mean` and `max`.
- All intermediates `phi` / `delta_t` / `score` / `normalized` / `head_score` / `kept` are returned for item-by-item assertions.
- The output `phi` is permuted from the internal `(L,T,H,F)` to `(L,H,T,F)`; `delta_t` is returned as float with shape `(T,O)`.
- Ties go to the smaller token index; upstream breaks ties with `tie_break_noise`.

## Known upstream differences

- The upstream `KV_BUDGET=2048` / `DIVIDE_LENGTH=128` / `WINDOW_SIZE=128` eviction flow is not implemented; this problem only scores and performs an explicit top-k.
- Upstream explicitly raises for GQA; this problem requires `q_mean` and `k` to agree on `H` and does not model group replication.
- The actual upstream RoPE tensor layout may differ from this problem's adjacent real/imag packing; here `k[..., 2f]` and `k[..., 2f+1]` form one pair.
- Upstream's `q_abs` is a per-`(l,h,f)` query-magnitude statistic; here it is supplied directly, and `extra` is 0 when `q_abs == |q_mean|`.
- Upstream z-normalizes, averages layers, then top-ks, same as here; the differences are mainly the budget, tie-breaking, and how many intermediates are exposed.

## Complexity

With \(O\) offsets (8 levels for the default 128 frames), scoring costs about \(O(LTHFO)\); z-normalization and top-k cost \(O(LHT + HT\log T)\). Compared with full attention's \(O(T^2)\), this is linear overhead.
