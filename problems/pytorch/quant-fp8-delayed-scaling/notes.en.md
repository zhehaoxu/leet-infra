# notes (quant-fp8-delayed-scaling)

## Source

- Upstream repo: `NVIDIA/TransformerEngine`, main branch. The old
  `transformer_engine/pytorch/fp8.py` is marked DEPRECATED; the logic now lives in
  `transformer_engine/pytorch/quantization.py`:
  - `_default_sf_compute(amax, scale, fp8_max, margin)`:
    `sf = (fp8_max / amax) / (2**margin)`, followed by three `torch.where` calls —
    `amax > 0` false → keep the old `scale`; `isfinite(amax)` false → keep the old `scale`;
    `isinf(sf)` true → `_fp32_max` (`torch.finfo(torch.float32).max`).
  - `_default_get_amax_and_update_history(amax_history, algo)`: `max` → window maximum,
    anything else (`most_recent`) → `amax_history[0]`, then `_update_amax_history`.
  - `_update_amax_history`: `torch.roll(h, -1, 0)` only when `shape[0] > 1`, then `h[0].fill_(0.0)`.
- Defaults (`DelayedScalingRequest`): `margin=0`, `amax_history_len=1024`, `algo="max"`.
  This problem shrinks the window to 1–8 for hand computation without changing any formula.
- `fp8_max`: 448 for E4M3 and 57344 for E5M2 (decided by `Format`); it is an explicit input here.

## Traps in the upstream implementation

1. **An `inf`/`nan` amax never shrinks the scale**: `torch.where(isfinite(amax), sf, scale)`
   falls straight back to the old scale. Once a cast overflow produces an `inf` amax, the scale
   stays put and the next step overflows again — official behaviour, not a bug, reproduced here.
2. **`amax == 0` keeps the previous scale instead of falling back to 1.0**: the docstring says
   "previous scale (or 1)", but the code only holds `scale` (initialized to 1.0); follow the code.
3. **The clamp is outermost**: the three `where` calls replace unusable branches with the old
   scale first and only then clamp `isinf → FP32 max`. So `amax == 0` (which computes `inf`)
   never clamps; only a "usable but tiny" amax does.
4. **Slot 0 is zeroed after the left roll**: reversing the roll direction pins one large amax
   in the window forever and keeps the scale too small.
5. **A window of length 1 does not roll but still zeroes**: `shape[0] > 1` gates the roll;
   the zeroing is unconditional, which degenerates to "only the current amax is kept".
6. **float32 narrowing**: `amax_history` and `scale` are float32 tensors, so `1234567.89` becomes
   `1234567.875` in the window; replicating in float64 mismatches in the last bits.

## Normalizations in this problem

- The ledger exposes both `scales` (used for the cast) and `next_scales` (after the update),
  plus `histories` snapshots, so the one-step lag and the window roll are observable instead of
  inferred from float drift.
- The two easy-to-confuse branches get separate boolean columns: `kept` is "amax unusable" and
  `clamped` is "usable but sf overflowed".
- Validation gates: `history_len` positive integer (`bool` rejected), `algo` whitelist, `margin`
  a non-negative integer, `fp8_max` / `init_scale` finite and positive; an empty sequence returns
  an empty ledger with `final_scale == init_scale`.
- Every scalar operation runs in float32 and is widened with `double()` on return, so results are
  bit-reproducible.

## Mutations

All 17 mutants are CAUGHT:

```
m-margin-mul                 margin used as margin+1     → margin check
m-kept-invert / m-kept-drop  fallback inverted / removed  → zero and inf checks
m-clamp-drop / m-clamp-value clamp removed / 1e38         → 1e-40 check
m-roll-dir                   roll direction reversed      → eviction check
m-no-zero-slot               slot 0 not zeroed            → histories assertions
m-no-roll                    never rolls                  → eviction check
m-algo-swap                  max / most_recent swapped    → comparison check
m-scale-recorded-after       scales records the new value → lag check
m-hist-fp64                  window in float64            → narrowing check
m-usable-finite-only         finite test without > 0      → amax=0 check
m-usable-positive-only       > 0 test without finite      → inf/nan check
m-validate-history/margin    two validation gates broken  → validation check
m-final-history-stale        final_history from old snap  → tail assertion
m-validate-fp8max            fp8_max gate relaxed to >=0  → validation check
```

## Overlap check

- FP8 round-trip (125) and E5M2 block scaling (445) cover element formats and static per-block
  shared exponents, with no cross-step recurrence.
- `quant-fp4-e2m1-grid`(740) / `quant-fp6-dual-subformat-pack`(741) cover element bit fields and
  packing.
- The only subject here is the scale recurrence state machine with its window and two fallback
  branches, which does not overlap any of the above.
