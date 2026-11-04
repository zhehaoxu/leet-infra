# Solution Notes: MagCache Accumulated-Error Gate Step

## Source Localization

- Source file: diffusers `hooks/mag_cache.py` (469 lines, MagCache paper/repo github.com/Zehong-Ma/MagCache). Core: `MagCacheConfig` (threshold=0.06, max_skip_steps=3, retention_ratio=0.2 defaults; raises when ratios missing and calibrate=False; `nearest_interp` when lengths differ, L110-137), `nearest_interp` (nearest-neighbour index interpolation, from L69), `MagCacheState` (previous_residual + three accumulators + step_index, reset clears all, L140-168), `MagCacheHeadHook.new_forward` (scale out-of-range fallback → retention_step → accumulation → decision → two output paths, L200-280), `MagCacheBlockHook` (is_tail writes previous_residual = tail output − head input, L330-360).
- The official skip output formula: `output = hidden_states + previous_residual` (L232/240, head input + last tail residual); compute runs the original forward and the tail hook writes the residual. The judgement domain compresses "tail output" into the tail_output argument — the residual formula `tail_output − hidden_states` is bitwise identical to the official one.
- The official decision happens in the head hook (before block execution), with the accumulators absorbing this step's contribution before the skip test; on skip the tail hook passes the input through (`return hidden_states`) and advances the step. So skip steps **keep** the accumulators and the compute-step reset completes inside the decision branch — the judgement domain's single function called per step is equivalent to the official "head decides + tail writes residual" split.

## Math Notes

1. **Structure of the accumulated error**: err += |1 − ratio| where ratio = Πscale (running product). Unlike the per-step "err += |1 − scale|": the accumulated contribution of 0.98² is |1−0.9604| = 0.0396, not 0.02. `check_accumulated_error` hand-computes three steps (0.02 → 0.0596 → 0.1184 breaching the threshold) against the implementation.
2. **Retention rounds half-up**: `int(r·N + 0.5)` is round-half-up on positives. The truncation mutant `int(r·N)` is observable only when frac(r·N) ≥ 0.5 — 0.2·8 = 1.6 (round 2 / trunc 1). **The key construction for the anti-truncation case**: the step inside retention must carry a non-None residual and assert the accumulators keep their initial values — with only `computed is True and steps == 0` asserted, under the truncation mutant the step falls into the accumulation zone, hits compute+reset because res is None, and steps is 0 again — the assertion is fooled by the reset path (exactly how the first draft SURVIVED).
3. **Accumulate first, decide second**: even when the decision must be compute (res None / err over / steps over), ratio/steps/err are already updated; only the compute branch resets to the initial values. The mutant m-skip-res (dropping the res None condition) catches "counting as skip even when res is None".
4. **err ≤ threshold with exact equality**: the official comparison is `<=` (equal still skips). The boundary uses powers of two: scale=0.5, ratio=1.0 → err=0.5 exactly equals threshold=0.5 → skip.
5. **Out-of-range scale fallback**: after interpolation len==N, but for step ≥ N (the judgement domain allows explicitly passing a large step) the official fallback is scale=1.0 — the ratio stays unchanged while err still accumulates |1−ratio|. The fallback is not "do nothing".

## Implementation Notes

- Validation order: tensor types/shape → step/N/threshold/retention/max_skip numeric domains → mag_ratios (None rejected, list converted, 1-D non-empty, floating point, length interpolation) → state structure. State validation checks **unknown keys before missing keys** (`{"foo":1}` reports unknown keys, not missing) and missing keys before value types.
- nearest_interp as a standalone function: src validation (1-D non-empty floating tensor) + target_length validation (not bool, ≥1) → target==1 takes the last element → scale/grid/round indexing.
- Functional state: new dict + clone; the compute branch residual is `(tail_output − hidden_states).clone()`, the skip branch keeps a clone of the old residual.

## Testing Notes

- Full-loop cross-check: an independent `_oracle_loop` (same state machine) generates the computed sequence.
- retention_rounding: two rounding combinations (0.2·8, 0.25·10) + **the anti-truncation combination (0.2·8's step 1 with a residual, asserting zero accumulation)** + accumulator retention on skip (ratio=1.5/err=0.5).
- accumulated_error: three hand-computed steps + reset assertions + skip output = hidden + old residual (written as the `(tail − hidden)` expression, not a literal — `1.0+0.6−1.0 ≠ 0.6` in floating point).
- max_skip_steps: the full sequence with scale=1 (err always 0) — steps 1,2,3 skip, the 4th compute+reset, then 1,2 skip.
- stale_residual: changing tail_output on a skip step does not affect the output; the compute step updates the residual.
- nearest_interp verified indirectly through solve (`import solution` is unavailable in the judge's separate namespace): err = |1 − interpolated ratios[step]| back-solves the index, covering round's banker's rounding (0.5→0), target==1, and list inputs.
- Validation-case caveat: `solve` has default parameters, so `_expect_err` must not duplicate explicit keywords with `**kw` (the first draft clashed `**kw` with an inline threshold).

## Mutation List (31, all CAUGHT, no equivalents)

Retention/gate 3: +0.5 truncated, >=→>, retention deleted. Accumulators 10: ratio assignment replacing the product, err using the per-step scale, err without abs, err <=→<, steps <=→<, res None condition dropped, err condition always true, reset deleted, reset ratio=0, scale out-of-range fallback 0.0. Output paths 6: skip output using tail, compute output using hidden, residual referencing itself/negated, skip-step residual not kept. nearest_interp 4: scale formula, round→floor, target==1 taking the head, interpolation deleted. Validation/messages 8: negative step, N≥1, negative threshold, retention range, max_skip≥1, ratios None message, integer ratios, missing state keys + error message.

Experience: ① "no accumulation inside retention"-style assertions must guard against the reset path's disguise — carry a non-None residual and assert the accumulator initial values; ② the compute-step output value also needs an assertion (testing only the skip path lets a compute-output substitution SURVIVE); ③ when the docstring and code share a string, locate mutation fragments with indentation/context; ④ `import solution` does not work in test_cases under the judge's separate namespace — indirect verification (observable side effects through solve) is the way.
