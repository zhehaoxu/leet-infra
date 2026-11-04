# Solution Notes: SeaCache Spectral-Indicator Gate Step

## Source Localization

- Source file: diffusers `hooks/sea_cache.py` (1097 lines, SeaCache paper arXiv 2602.18993, Cosmos3/Wan video transformers). Core: `_apply_sea_filter` (three-axis Wiener-style spectral gain + density normalization, L362-389), `_build_indicator` (filter after `latent.movedim(0,-1)`, L392-405), `SeaCacheSharedState.resolve_gate` (accumulated relative-L1 budget + five forced-compute conditions + non-adjacent reset + (step,sigma) memo, L182-245), `SeaCacheRootHook.prepare_decoder_stack` (Cosmos stack path: residual reuse/extrapolation/two fail-opens, L606-724), `SeaCacheLeaderBlockHook.new_forward` (the same for the generic path, L790-878), `_record_full_execution` (history truncated to residual_order+1, L288-311).
- The official two entries (the Cosmos `_run_decoder_stack` stand-in / the generic leader block hook) share `resolve_gate`; both skip outputs are `hidden + cached_residual`. The Cosmos path clears only history on residual shape mismatch, the generic path clears both — the judgement domain takes the generic semantics (zeroing).
- The official gate memo `gate_key == (step, sigma)` returns the cached decision directly: an idempotence guard for the root+leader double entry within one forward; the judgement domain's single call has no such need.

## Math Notes

1. **Shape of the Wiener-style gain**: `axis_gain = s·cp/(s²·cp + n² + 1e-16)`, s=1−σ, n=σ, cp=1/(|f|^p+1e-16). At low frequency cp→1e16 so axis_gain→1/s (signal boost), at high frequency cp→0 so →0 (noise suppression) — a classic denoising gain. The three axes multiply and then divide by the mean (density normalization) to keep cross-step distances comparable.
2. **The T=H=W=1 identity**: fftfreq(1)=[0], cp=1e16, axis_gain≈1/s, product (1/s)³, normalized to exactly 1 — the indicator ≈ identity (float32 error ~1e-7). This gives sea_filter an oracle-free quick sanity test.
3. **The `>=` semantics**: `candidate >= threshold` computes. threshold=0 plus a zero distance (bitwise-reproduced same latent, same sigma) builds the exact boundary candidate=0.0 — the `>` mutant would wrongly skip.
4. **distance references the filtered previous**: `num/den`, den=mean|prev|+1e-16. Note in float32 `den + 1e-16 == den` (a no-op), and abs commutativity makes swapping current/prev unobservable in the numerator — **the denominator reference (prev vs cur) is the observable mutation**, and a delta=0.3 latent pair gives a ~9% deviation (delta=0.02 gives 3.7e-9, unobservable — mutation-case deltas must be large enough).
5. **The extrapolation scale**: `(step−latest_s)/(latest_s−prev_s)`. Adjacent history (step gap 1, current step +1) is constantly 1 — only non-adjacent history (steps 0 and 2, current step 3 → scale=0.5) catches the numerator/denominator swap.
6. **The non-adjacent reset precedes all gating**: after reset previous_indicator=None → first observation forces compute → distance=None. A backward jump (last=3 → step 1) is also non-adjacent — `!=` must not weaken to `>`.

## Implementation Notes

- sea_filter is a standalone module-level function (official同名 semantics); power_exp/sigma validation lives inside sea_filter (**do not duplicate the power_exp check in solve** — duplicate guards shield each other and single-copy mutations become uncatchable).
- solve validation order: three tensors → latent 4-D → gen shape → step/N → sigma domain → threshold → residual_order ∈ {0,1} → retention/cache_end/max_consecutive → state five keys (unknown before missing, history entry whitelist).
- History entries `{"step", "residual"}`; append then truncate `[-(residual_order+1):]`; skip steps keep history and increment consecutive; compute steps zero consecutive.
- Functional: `list(state["history"])` shallow copy + new entries; previous_indicator stored as `.detach().clone()`.

## Testing Notes

- Three layers of indicator verification: T=H=W=1 identity (oracle-free) → independent numpy float64 oracle cross-check (atol 1e-4; float32/64 difference ~1e-6) → sigma=0/1 clamp endpoints.
- Exact boundary: threshold=0 + zero distance (bitwise-deterministic same latent/sigma) catches `>=`→`>`.
- **Distance values need a relative tolerance** (`< 1e-3·d`): the absolute error of a float32 implementation against a float64 oracle grows with d, so an abs tolerance is either too loose (missing the denominator mutation) or too tight (floating noise flips it); a delta=0.3 latent pair makes the denominator-mutation deviation ~9%.
- Full loop: a 6-step oracle state machine cross-checks the computed sequence, after self-checking that every oracle distance sits ≥30% from the threshold boundary (`d < 0.7·thr or d > 1.3·thr`) so float32/64 differences cannot flip decisions.
- Adjacency: the cache_end case must walk adjacent steps into the window (jumping straight to step 3 triggers the non-adjacent reset first and never tests the window); the extrapolation-scale case must insert a skip step to keep adjacency.
- Both fail-open channels + indicator shape mismatch (distance=inf) + non-None residual shape/dtype mismatch + purity (**snapshot history per entry: step + torch.equal residual** — `list[dict] ==` dies on ambiguous tensor truthiness).
- State-validation assertion substrings should avoid the `state['key'] 必须...` quote boundary (match on the key name or a prefix-free phrase).

## Mutation List (33, all CAUGHT; plus 6 identified as equivalent and dropped)

sea_filter 7: signal/noise swap, power_exp exponent dropped, clean_power's 1e-16 dropped, three-axis multiply→add, density normalization dropped, .real→.imag, movedim dropped. Gate 12: >=→>, retention <→<=, cache_end >=→>, max_consec >=→>, max_consec condition dropped, non-adjacent !=→>, reset keeping the indicator, distance numerator mean→sum, denominator referencing cur, shape check dropped, accumulated not zeroed. Residual 9: no-history fail-open dropped, shape-mismatch fail-open dropped, extrapolation condition order flipped, scale numerator/denominator swapped, extrapolation negated, skip output using gen_output, residual negated, consecutive +2, compute not zeroing. Validation 5: negative step, sigma domain, order set, power_exp domain, state missing keys + message.

**Equivalent-mutation record (dropped)**: ① sigma clamp endpoint removal — the 1e-6 deviation is below any feasible float32 tolerance; ② denominator +1e-16 removal — a float32 addition no-op; ③ record-side history truncation removal — the read-side slice already covers it; ④ `.contiguous()` removal — semantically a no-op; ⑤ return dtype `.to(dtype)`→`.to(float32)` — judgement-domain inputs are all float32; ⑥ numerator current/prev swap — abs commutativity.

Experience: ① redundant double guards (solve and sea_filter both validating power_exp) make single-copy mutants forever SURVIVED — remove the solve-side copy so the guard is unique; ② the observability of numeric mutations must be measured on real data (3.7e-9 and 6e-2 are separated by one delta choice); ③ a relative tolerance is the right form for the "float32 implementation vs float64 oracle" setting; ④ semantics involving step adjacency (non-adjacent reset, extrapolation scale) require test step sequences that advance legally, otherwise another branch preempts.
