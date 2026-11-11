# Authoring notes: 869 = au-torchaudio-biquad-coeff-ledger

## 1. Topic selection and boundary

The `functional.py` sourced by order 855 (2511 lines / 55 top-level functions) is fully covered by orders 855-868, so the source was switched to `pytorch/audio main::src/torchaudio/functional/filtering.py` (1702 lines / 28 top-level functions, fetched from GitHub raw on 2026-11-10).
Duplicate check: the nine coefficient functions plus `_dB2Linear` have ZERO hits across all `problems` meta / description / notes; however order 861 ALREADY covers `biquad`, `lfilter`, `_lfilter`, `_lfilter_core_generic_loop` and the K-weighting IIR recursion. Order 869 therefore judges coefficient derivation only, with `lfilter` injected as a spy. Excluded: `filtfilt` / `flanger` / `phaser` / `overdrive` / `dither` / `dcshift` (RNG / wavetable / loops), `gain` / `contrast` (too thin) and `vad` (overlap risk with `asr-fsmn-vad-endpoint`).

## 2. Verbatim official environment

`official869.py` extracts 11 functions with `ast.get_source_segment` into one shared namespace without modifying any body; it injects real `torch` (float64), `math`, `warnings`, `typing`, `F` and `_IS_TORCHAUDIO_EXT_AVAILABLE=False`. The only substitute is the `lfilter` spy, which records `waveform_shape` / `a_coeffs` / `b_coeffs` / `clamp` / `batching` / `dtype` and returns the waveform unchanged. No RNG, no audio IO, no warning.
Smoke `SMOKE 869 DONE`: lowpass / bass coefficients, `bass a0_raw=1.0`, call clamp/batching/dtype, six preset rows, seven error cases (deemph/riaa illegal rates raise ValueError, Q=0 / sr=0 do not), the biquad_order coefficient ordering, and the grid over nine kinds / 26 rows all pass.

## 3. Key mechanics

`w0=2*pi*f/sr`, `alpha=sin(w0)/2/Q`, `A=exp(gain/40*ln10)`; the SoX `temp1/temp2/temp3` scaffolding. `bass` divides ALL SIX coefficients by `a0` before calling `biquad` (a0_raw=1.0) while the others pass unnormalized coefficients; `equalizer` is asymmetric between `alpha*A` and `alpha/A`; `band` uses `bw_Hz=f/Q`, `a2=exp(-2*pi*bw_Hz/sr)`, a `mult` branch for `noise=True` and `b1=b2=0`; `bandpass` has `const_skirt_gain`; `deemph` (44100/48000) and `riaa` (four rates) use fixed tables plus illegal-rate ValueErrors, and RIAA additionally normalizes with `g` from the real/imaginary parts of the 1 kHz response; `biquad` puts a_coeffs first and b_coeffs second.

## 4. Verification pipeline

The 12-segment mechanism probe `probe869.py` reports **485/485 ALL GREEN**: each segment recomputes a/b with an independent closed form and cross-checks z-domain transfer-function properties (all-pass magnitude, unity DC gain, notch, peaking gain A^2, bass shelf DC=A^2, 1 kHz normalization); it also pins the 14 census source facts, the preset tables, illegal-rate messages and the IEEE boundaries for Q=0 / sr=0 / negative rates / band Q=0.
The layer-2 differential `probe_solution869.py` reports **91419 checks / 2161 configs / 0 bad**: official `comp` versus reference `solve` over 9 kinds x 5 rates x 5 frequencies x 5 Q values, bass/equalizer x 5 gains, band x noise, bandpass x skirt, presets, biquad, boundaries and five waveform shapes, with exception types and messages aligned as well.
The three-layer independent oracle `verify869.py` reports **26832 checks / 475 configs / 5679 algebraic properties / 0 fails**: trig via `cmath.exp(1j*x)`, `A` via `10**(gain/40)`, `exp` via `cmath.exp`, `sqrt` via `x**0.5`, pi via `4*atan(1)`; RIAA polynomials rebuilt with exact `Fraction`; plus structural cross-checks (call / presets / grid / errors / census / db2linear / biquad_order against coeffs, fresh census dict, unknown-mode message).

## 5. Mutation and equivalence evidence

`mut869.py` holds 208 mutants. The first round gave 193 CAUGHT / 10 SURVIVED / 4 ERROR; the four ERRORs were defects in the mutant definitions themselves (two occurrences of `t1 = 2*_sqrt(A)*alpha` in deemph and bass, two `waveform_shape` anchors in call and coeffs, and a multi-line census replacement breaking indentation), fixed with explicit occurrence indices and full-block replacements, plus one added `ca-shape-coeffs` mutant.
Of the ten survivors, FIVE were JUDGE BLIND SPOTS, captured in two hardening rounds. Round one: `ie-div-sign` (negative sample rate with `Q=0` yields `-inf`), `ie-exp-inf` (`band` with `Q<0, sr=0` yields `+inf`), `ie-shape-msg` (ragged waveform message) and `ca-clean` (`call` must canonicalize non-finite coefficients). Round two followed the first `equiv869.py` run, which showed `ie-sqrt-neg` was NOT equivalent: with `band`, negative Q and `noise=True`, the noise-branch discriminant is a FINITE NEGATIVE number, where official `torch.sqrt` returns `nan` but Python `math.sqrt` would raise `ValueError`. It was the fifth blind spot, captured by `check_band_noise_negative_discriminant`. Fixtures grew from 117 to 132 constants and checks from 49 to 53.
The other FIVE are SEMANTICALLY EQUIVALENT, proven by `equiv869.py` with 13128 configs x 9 modes = 118152 comparisons PER MUTANT, 590760 comparisons total, comparing the FULL `solve()` output (including exception type/message and float bit equality) with zero differences: `cf-args-bass` / `cf-args-biquad` because ARGS feeds only a dict comprehension (dict equality ignores order), `ie-err-type` because `_err` is reachable only from `errors` whose two raising cases are both ValueError, and `ri-b-re` / `ri-a-re` because cos is even and `math.cos(-y)` is bit-identical to `math.cos(y)`. NO mutant is deleted: 208 mutants minus 5 genuine equivalents leave 203 active, ALL CAUGHT / 0 SURVIVED / 0 ERROR.

## 6. Engineering lessons

(1) torch tensor division by zero returns `inf/nan` while Python `math` raises, so the reference must emulate it with `_div` / `_sqrt` helpers; (2) `-inf` / `nan` are not valid Python literals, so fixtures must canonicalize them to stable strings or the judge file cannot be parsed; (3) when one anchor appears in several functions, the mutation script must locate it by explicit occurrence index or the item reports ANCHOR-ERROR; (4) mixed-sign inf (`Q=0` with a negative sample rate) distinguishes a bare `float('inf')` from sign-correct inf and is the key case for hardening the IEEE layer.

Land 11 files, selftest in both modes, the bilingual check, register 847 problems, short guard and full guard with an isolated `--basetemp`, then backfill the log. No git commit until the user confirms.
