# Implementation notes: torchaudio BS.1770-4 loudness-gate ledger

## 1. Topic selection and deduplication

Order 861, collection `audio` 12 to 13, slug `au-torchaudio-loudness-gate-ledger`. A library-wide scan showed `max_order=860`, 838 problems, unique orders, 861 free and the slug unused; `loudness`, `LKFS`, `BS.1770`, `biquad` and `lfilter` have no hits in any `meta.json`, and the only description/notes match is one candidate-avoidance sentence in 860. It shares the `functional.py` / `filtering.py` ecosystem with 855 (spectral post-processing), 857 (waveform operators), 858 (NCCF), 859 (sinc resampling) and 860 (Griffin-Lim) but uses a completely disjoint set of functions.

## 2. Verbatim execution environment

`functional.py` is the 2511-line official file downloaded for 855; `filtering.py` was fetched from the GitHub raw endpoint on 2026-11-08 and its nine functions/classes transcribed verbatim into `filtering861.py` (only the module-level `_IS_TORCHAUDIO_EXT_AVAILABLE` switch is supplied by the execution environment, no function body is touched). `official861.py` uses `ast.get_source_segment` to extract `loudness` plus the eight filtering definitions, `exec`s them into a shared `NS`, then binds `NS['_lfilter_core_loop']` to `NS['_lfilter_core_generic_loop']`.
Smoke output at sr=8000: treble `a=[3.625826704781014, -1.2110558364958972, 0.6938511881834482]`, highpass `a=[1.0298406997386809, -1.9991093343177666, 0.9701593002613191]`; a sine through both biquads gives `gate_samples=3200`, `step=800`, `n_blocks=3`, `LKFS=-9.97723766646316`.

## 3. Pipeline results

```text
12-segment probe      PROBE 861 DONE ok=261 fail=0 ALL GREEN
three-layer verify    VERIFY 861 DONE checks=931432 fails=0 ALL GREEN
judge cases           DRIVER 861 50/50 ALL ACCEPTED
mutation             199 total / 6 equivalent / 193 CAUGHT / 0 SURVIVED
precheck              PRECHECK 861 35 PASS / 0 FAIL
```

The independent oracle of the three-layer check computes `A` with `math.pow(10, gain/40)`, rewrites FIR + IIR using explicit `xp=[0,0]+x` / `yp=[0,0]+zeros` padded arrays with `math.fsum`, and recomputes the gating ledger with a block-first loop; it agrees value by value with both the verbatim official environment and the pure-Python reference. The 931432 count is the number of recursive scalar comparisons, including element-wise checks over 16000-sample waveforms.

## 4. Mutation testing and true equivalents

First round, 197 mutants: 177 CAUGHT, 17 SURVIVED, 3 ANCHOR-ERROR. The anchor errors were three repeated strings lacking occurrence indices (`return out` appears in both `_coeff_block` and `_filter_row`, `return led` in both `_dual_gate` and `_mode_loudness`, and the 8-space `must be rectangular` is also a substring of the 12-space version); adding `idx` fixed them. The hardening patch adds eight cases: F7 (-69.565 LKFS between -70 and -60), F8 (`T == gate_samples`, exactly one block), F9 (finite energy but every block below -70), F10 (five blocks dropped by the absolute gate yet still above `gamma_rel`), F3L (sr=35 banker's rounding step=4), the sr=1 error message, the default clamp, and empty-coefficient messages; F5 also gains `n_gated_abs` / `ew_abs` assertions and R2/R3 gain multi-channel `shape` assertions.
Six mutants were ruled true equivalents with evidence: (1) `MODES` is a documentation constant appearing exactly once (its definition); (2) `math.log10(nan)` already returns nan, so removing the isnan early return is equivalent; (3,4) the three call sites of `_log10` can only receive energies (sums of squares times non-negative gains) or nan, so the negative branch and its return are dead code; (5,6) `require_matrix` guarantees `filtered` is non-empty, making the `len(filtered)<1` branch unreachable. Final result: **193/193 CAUGHT, 0 SURVIVED, 0 ERROR**.

## 5. Scope reductions and engineering lessons

(1) The judged domain takes only two-dimensional `(channels, time)`; the official batch axis is stripped and the 0-dimensional LKFS tensor becomes a float. (2) `blocks` takes an already filtered waveform so the gating ledger can be judged independently of the biquads. (3) float32 and the C++ extension paths are recorded but not judged; numerics are float64. (4) `pad`, `batching` and the 2-D coefficient branch of `lfilter` are outside the judged domain. (5) The judge helper `_near` must recursively `assert` rather than return a bool (the lesson of order 860, applied here from the start). (6) Waveforms are only run at `sr>=4000`: below that the 1500 Hz shelf has poles outside the unit circle and both torch and Python overflow, while the coefficient mode remains judgeable.

## 6. Official quirks and deduplication conclusion

Same as section 8 of `description.md`: the size(-2) channel check, two spellings of alpha, the exp form of A, clamp not feeding back, one clamp per stage, the dtype of g, the shape of count_nonzero, strict -70, gamma_rel consuming only absolutely gated blocks, logical_and, nan when all blocks are dropped, four literal constants, and no Nyquist check. One correction to the notes of 860 is needed: `loudness` has no call relationship with `phase_vocoder` at all.

## 7. Registration and guards

`register861.py` registers with minimal intrusion, growing `audio` from 12 to 13 and appending the 861 convention paragraph to both descriptions; verification expects `839 registered = 839 disk / max_order=861 / no dangling / no unregistered`. `todos.md` is updated to 839 problems / audio 13 / next batch from 862. The library-wide bilingual check expects `0 error / 15 warning` (all 15 historical); the short guard uses an isolated `--basetemp`, and the full guard result is backfilled after its background run.
