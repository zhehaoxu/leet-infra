# Reference Solution: Sort Direction, Budget Value Passing and Flattened Block Indices

## Implementation

1. Validation: documents non-empty, `doc_ppl` equal-length and finite, `token_loss` total-length aligned, `context_budget` regex, enum whitelists, force indices in range.
2. **Ranking**: `direction = -1 if condition_in_question == "none" else 1`, **stable** sort by `direction * ppl`. Upstream `get_distance_longllmlingua`'s score is `get_condition_ppl(d, query + " We can get the answer to this question in the given documents.") - dl * 2 / 250 * 0` — the length term is multiplied by 0; here it arrives as `doc_ppl`.
3. **Budget**: `target<0 → 100`, then one arithmetic step keyed by the first character of `context_budget` (upstream `eval("target_token" + context_budget)`, narrowed here to a regex whitelist). **Key point**: the variable decremented during selection and the numerator of `tau` in the window stage are **two different values** — upstream `compress_prompt` passes the evaluated `target_token` unchanged into `iterative_compress_prompt`, and the decrements inside `control_context_budget` live only in a local. The solution keeps `budget` (decremented, stop condition only) apart from `target_token` (feeds tau).
4. **Selection**: walk the ranked order subtracting the **raw length** (no +2; +2 only enters the window ledger), `force_context_ids` seeded, later hits deduped; stop when `budget < 0` (the overshooting document is still selected). `two_stage` splits the selection order by **position** parity and reverses the odd segment — the parity is of the used list's positions, not of document indices.
5. **Dynamic ratios**: `[i * λ/(N-1) for i in range(-(N-1), N, 2)][::-1]`, 0 when N=1; `abs(λ)`; a dict keyed by document index carries ratios through the reorder. λ=0 short-circuits to zeros (numerically identical to the map path).
6. **Window plan**: `L = len + 2` (different ledger from selection's raw length), `tau = budget / (ΣL + 1)`, `r = clamp(tau + dr, 0, 1)`; `last` starts at 1 (virtual BOS); `last + L >= W` closes a window with `(W - last, r)`, the remainder factors out k whole `(W, r)` windows and its residual becomes the next window's head, a zeroed remainder leaves nothing.
7. **Thresholds and masks**: the g-th flattened pair covers `[gW, gW + delta_end)` (upstream advances `end` by `iterative_size` per pair; derivation below); the threshold is computed over the **whole stream** loss with 10000 sentinels filtered, descending index `clamp(int(len*ratio)-1)`, `ratio==1.0 → -inf`; masks keep `loss > thr` inside the block, position 0 always, everything outside always.

**Derivation of the flattened block index** (the subtlest point here): `get_compressed_input` filters `[end_arg - delta_end, end_arg)` where `end_arg = end - W + delta_end`, so the region is always `[end - W, end - W + delta_end)`; `end` starts at `min(W, N)` and gains `W` after every pair, hence the g-th pair's region is exactly `[gW, gW + delta_end)` — **counted per pair, not per window** (pairs of a multi-pair window land in consecutive distinct blocks). This is first-round, single-stream semantics: real upstream execution contracts the stream across windows (`end -= dropped`) and reshuffles `past_loss`; this problem produces a static plan only, see the deviation notes at the end.

## Correctness Checks

6 `check_*` functions cover: the worked example (full three-layer chain asserted field by field), 8 multi-knob oracle cases (question direction, force, `*2`/`/5` budgets, negative target → 100, sentinel, N=1 zero ratios, single document hitting ratio 1.0, a 5-document two_stage), budget exhaustion and force consuming budget, the k-whole-windows + remainder plan for an oversized document, threshold direction and sentinel (equality dropped, sentinel kept but excluded from statistics), and 18 invalid inputs.

The oracle (`test_cases.py::_oracle`) is a reimplementation on different machinery: numpy `argsort(kind="stable")`, an `operator` dict for the arithmetic, a **remaining-capacity window simulation** (`capacity = W - last` view) instead of the upstream `last` cursor, and numpy sorting for thresholds.

Mutation testing (10/10 caught): flipped sort direction, bypassed `target<0→100`, tau fed the decremented budget, length without +2, ratio table not reversed, ratios not remapped by document, ascending threshold, flipped keep direction, block index counted per window (the critical mutant that only multi-pair windows expose), two_stage without reversing the odd segment (only detectable with ≥2 odd-position selections, hence the dedicated 5-document case).

## Scope and Sources

Upstream microsoft/LLMLingua (2310.06839, LongLLMLingua lives in the main repo):

- `prompt_compressor.py::get_rank_results` (L1818+) and `get_distance_longllmlingua` (L2032-2045): `sort_direct = -1 if condition_in_question == "none" else 1`, stable `sorted(enumerate(...))`.
- `control_context_budget` (L1173-1241): `eval("target_token" + context_budget)`, subtracting `context_tokens_length` (raw lengths) in ranked order, `force_context_ids` seeding, original/two_stage reorder, the dynamic ratio table and `dynamic_ratio_map` remap.
- `get_dynamic_compression_ratio` (L1023-1065): `L = get_token_length + 2`, `tau = target/(Σ+1)`, the `last` cursor, the k-whole-window expansion.
- `get_estimate_threshold_base_distribution` (L1508-1521): `ratio==1.0 → -inf`, `ppl[ppl != 10000]`, `int(len*ratio)-1` truncation, descending lookup.
- `get_compressed_input` (L1402-1506): `need_idx = loss > threshold` (keeps high NLL), `need_idx[end:] = 1` / `need_idx[:end-iterative_size] = 1` (only the current window moves), the BOS-shifted concat `loss[:1] > 0`.

Deliberate contract differences:

- **No model**: document conditional PPL arrives as `doc_ppl`, per-token NLL as `token_loss` (upstream `loss` is next-token aligned with length N-1; here it is token-aligned with length N and the BOS rule becomes "position 0 always kept").
- The concatenated stream inserts **no `\n\n` separator** (upstream joins with `"\n\n"` before tokenizing, so separators consume tokens; here documents are concatenated directly).
- `context_budget` is not `eval`ed — a single arithmetic step matching `^[+\-*/]\d+(\.\d+)?$`; `force_context_number` is not accepted (upstream's stop branch is dead code because `res` is never filled inside the loop, see L1202 and L1216-1219; dead code does not make a good exam).
- **Static-plan semantics**: upstream computes each pair's threshold over the contracted stream's `past_loss`, with `end` rewinds and KV eviction across windows; here thresholds are always computed over the **original** stream and masks are stamped on original indices — each pair is an independent first-round application. Cross-window contraction is out of scope.
- `condition_compare` / `keep_split` / `start` / structured segmentation (`get_structured_dynamic_compression_ratio`) are out of scope.
