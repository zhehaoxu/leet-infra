# Solution Notes: First Block Cache Residual Gate Step

## Source Localization

- Source file: diffusers `hooks/first_block_cache.py` (258 lines, ParaAttention's First Block Cache; the docstring self-describes as simpler than TeaCache, arXiv 2411.19108). Core: `FBCHeadBlockHook.new_forward` (first-block residual + gate dispatch + two output paths), `_should_compute_remaining_blocks` (absmean relative-L1 comparison), `FBCBlockHook` (middle/tail pass-through under a cache decision).
- The official shared `FBCSharedBlockState`: head_block_output, head_block_residual, tail_block_residuals (tail relative to the head **output**), should_compute (in-step transient). `reset()` clears only tail and should_compute, **keeping both head entries** — the judgement domain makes all three independently None-able.
- The official cache path synthesizes the output inside the **head hook** (tail_residual + current head output), with middle/tail blocks passing through — so compressing one forward into head/tail tensors is mathematically equivalent for the judgement domain.

## Math Notes

1. **Structure of the relative L1**: diff = mean|residual − prev| / mean|prev|. Numerator and denominator share the same aggregation, so any "same-aggregation substitution" is an identity ratio (mean→sum is strictly equivalent: sum = N·mean, and power-of-two scaling is bitwise exact in floating-point division — this problem's only equivalent mutation, dropped and recorded).
2. **Necessity of abs**: when the residual shrinks (common late in denoising), the residual difference is entirely negative, and an abs-less diff is negative → always ≤ threshold → always caching (wrong). `check_decreasing_residual` targets this with residual (0.15, 0.05) against prev (0.2, 0.1).
3. **Stale comparison**: cache steps write no state; the diff reference freezes at the last compute — accumulated drift across consecutive cache steps only recovers compute past the threshold. Asserted by a two-step scenario (small drift caches, large drift computes, independent of step A).
4. **The tail residual's reference**: `tail_block_residuals = tail_output − head_block_output` (this compute's head **output**), not head_input — mutation m-tail-res catches exactly this.
5. **Zero denominator**: with an all-zero prev residual the official nan/inf behaviour is normalized to 0.0/inf (nan>x False ≡ 0.0 ≤ threshold cache ✓; x/0 inf ≡ inf compute ✓).

## Implementation Notes

- Validation order: three tensor types → shape agreement → threshold (not bool, numeric, ≥0) → state structure (three-key whitelist, tensor-or-None values).
- The cache branch rejects a missing tail explicitly (official TypeError; judgement domain ValueError uniformly) — officially after reset (head kept, tail cleared) the first step does not force compute since residual is not None, and a small drift would hit the cache branch and crash officially. A genuine corner of the official reset semantics.
- Functional state: new dict + clone, with the purity case asserting the input snapshot unchanged.

## Testing Notes

- First step: computed=True, diff=None, state filled (tail_residuals = tail − head bitwise).
- Both compute/cache paths + diff value checked against an independent `_absmean_np` oracle.
- **Boundary cases must use exact powers of two**: prev (0.5, 0.5), difference (0.25, 0.25), thr=0.5 → diff exactly equals the threshold → cache; +2⁻⁴⁰ → compute. The first draft built "exact equality" from 0.21−0.2-style inexact numbers and floating-point error broke it (0.21−0.2 = 0.00999…98), making the `>=` mutation uncatchable — **boundary cases require binary-exact numbers**.
- Negative-difference case (abs necessity), both zero-denominator branches, two-step stale, per-entry unchanged state on cache steps, three-entry update on compute steps, purity.
- Residual construction uses the explicit two-column `_r(a, b)` helper throughout — `_t`'s linear second-column pattern (v·0.5−0.3) turns residuals into unintended values (the first draft's 5 failures came from exactly this); likewise `1.0+0.6−1.0 ≠ 0.6` in floating point, so expected residuals must be written as `(ho2 − hi)` rather than literals.

## Mutation List (26, all CAUGHT, no equivalents)

Decision core 6: residual referencing tail_output, numerator without prev, denominator using the current residual, >→>=, >→<, first-step diff non-None. Formulas 7: cache adding head_input, tail residual referencing head_input/itself, head_block_output using tail, residual cache not updated (stale broken), compute final output using head_output, first-step force flipped. Zero denominator 2: all-cache, all-compute. Aggregation 2: mean→max, abs removed. Defense/validation/messages 11: first-step cached, missing tail not rejected, shape removed, threshold negative/bool, unknown state keys, value types, three error messages.

**Equivalent-mutation record**: mean→sum is strictly equivalent under the diff = num/den ratio structure (same aggregation on both sides, sum = N·mean, bitwise-exact floating multiply/divide for power-of-two N) — dropped and recorded rather than forcing a case.

Experience: ① boundary assertions (>= vs >) need binary-exact construction of "exactly equal"; ② signed aggregations (abs removal) need a negative-difference scenario to be observable; ③ aggregation substitutions in a ratio structure may be identity equivalences — identifying and dropping them is more honest than forcing cases; ④ residual/output construction uses explicit per-column helpers, banning linear patterns and floating-point non-invertible expressions (`a+b−b ≠ a`).
