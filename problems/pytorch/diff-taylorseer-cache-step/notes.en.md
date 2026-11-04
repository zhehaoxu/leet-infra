# Solution Notes: TaylorSeer Cache Step

## Source Localization

- Source file: diffusers `hooks/taylorseer_cache.py` (348 lines, TaylorSeer paper arXiv 2503.06923, 2025-03). Three core pieces: `TaylorSeerCacheHook._measure_should_compute` (gating), `TaylorSeerState.update` (divided-difference recursion), `TaylorSeerState.predict` (Taylor extrapolation).
- A defensive branch worth noting in the official `update`: on a shape change it breaks and restarts the expansion (the comment explains Qwen-Image's KV prefill step returns prefix+target tokens whose shape differs, so differencing is impossible) — observable behaviour the judgement domain must cover.
- The official `_measure_should_compute` increments `state.current_step` on every call (hook-internal); the judgement domain makes step explicit and the state keeps only taylor_factors and last_update_step.
- The official factor dtype defaults to bfloat16 (`taylor_factors_dtype`); the judgement domain fixes float64 and drops the dtype parameter.

## Math Notes

1. **The difference table is a derivative approximation**. `f[j+1] = (f[j]_new − f[j]_prev) / delta` is multi-step numerical differentiation: f[1] is the first divided difference (≈ first derivative), f[2] the second (≈ second derivative), and delta is the compute-step gap rather than 1 — the biggest semantic difference from per-frame differencing (the m-delta-step mutation targets it).
2. **Taylor extrapolation**. `out(s) = Σ_j offset^j/j!·f[j]` with offset = current predict step − last compute step. max_order=1 degenerates to linear extrapolation f0 + offset·f1; max_order=2 adds the curvature term offset²/2·f2. With interval 5, offset reaches 4 and the curvature term contributes at the 8·f2 scale — not a small correction.
3. **The gating arithmetic**. `(step − before − 1) % interval == 0` means "aligned starting interval steps after warmup": with before=3, interval=5 the compute steps are 0,1,2,4,9,14 — and **step 3 is a predict** ((3−4)%5=4). Both off-by-one directions (subtracting before or not, the extra −1 or not) shift the compute steps observably; a full-loop computed_steps assertion catches either.
4. **Cooldown overrides the interval**. The after condition is an or, so every cooldown step computes (even if misaligned with the interval), and update continues (the recursion delta returns to 1).

## Implementation Notes

- Functional state: return a new dict, clone tensors, never mutate the input (the official StateManager mutates in place) — documented as a judgement-domain convention, asserted directly by the purity case.
- Validation order: outputs → step (int not bool, ≥0) → interval (≥1) → before (≥0) → after (non-negative or None) → max_order (≥1) → state structure (key whitelist, factor keys non-negative int orders, tensor values, last_update_step non-negative int or None). Missing the `-2` check for after was a first-draft bug (the "非负整数" message fooled the author).
- The predict raise condition is `last_update_step is None or not factors` (an or) — official semantics raise on both inconsistent directions; checking only factors misses lus=None with non-empty factors (m-predict-lus SURVIVED the first round for exactly this; caught after adding the inconsistent-state case).
- The compute-step delta==0 raise sits after the first-update check (first update has lus=None and is unaffected).

## Testing Notes

- **Full-loop comparison** (before=3, interval=5, max_order=2, steps 0..11): the oracle independently implements gating+recursion+extrapolation; per step assert the computed flag, the factor set and values on compute steps, and the output on predict steps — gating off-by-ones and recursion bugs get covered simultaneously.
- The interval-alignment case asserts the computed_steps sequence `[0, 2, 5, 8]` directly (before=1, interval=3) — note with before=1 the compute steps are 2,7,12, not 5,10 (the first draft assumed [0,5,8] and was caught by the baseline run).
- The cooldown case asserts `[0,1,2,4,6,7,8]` (every step computes once after=6).
- Hand-built state fed directly to predict (skipping recursion): first/second-order extrapolation checked against the numeric formula (`f0 + offset·f1 + offset²/2·f2`).
- Shape-change restart: feed a differently-shaped output on a compute step, assert only order 0 survives.
- State consistency in both directions: empty factors with lus set, and lus None with non-empty factors — predict raises on both.
- Purity: two identical calls agree, and the input snapshot is unchanged.

## Mutation List (31, all CAUGHT, no equivalents)

Gating 8: warmup boundary loosened, warmup removed, interval off-by-one both ways, interval removed, cooldown boundary tightened, cooldown removed, or→and. Recursion 6: delta0 check removed, delta using step, max_order−1, shape-restart condition removed, difference sign, missing delta division. Extrapolation 6: offset using step, offset constant 1, factorial +1, power +1, sum sign, lus check removed. Validation/messages 11: step negative/bool, interval 0, max_order 0, before negative, after negative, factor value non-tensor, lus bool, unknown state keys, delta message, predict message.

Experience: asserting the computed_steps sequence for gating off-by-ones has far higher coverage density than point assertions; both directions of inconsistent state (empty factors / lus None, and each combined) need explicit cases — the or structure of the official raise condition means removing either branch requires a dedicated case to catch it.
