# torchaudio biquad coefficient ledger: nine RBJ/SoX cookbook filters, the bass pre-normalization, the band exp/sqrt/noise branch, the de-emphasis and RIAA preset tables, and the a-b ordering handed to lfilter

## 1. Background

Nine `*_biquad` functions in `torchaudio.functional.filtering` each turn an analog prototype into second-order IIR coefficients via the bilinear transform, then all call `biquad(waveform, b0, b1, b2, a0, a1, a2)`. That function passes `torch.cat([a0,a1,a2])` as `a_coeffs` and `torch.cat([b0,b1,b2])` as `b_coeffs` to `lfilter`. This problem judges ONLY the coefficient derivation, reproducing it in pure Python as eight independently judged modes (`coeffs` / `call` / `db2linear` / `presets` / `errors` / `census` / `grid` / `biquad_order`) and pinning every value to verbatim official snapshots.
Library-wide duplicate check: the nine coefficient function names and `_dB2Linear` have ZERO hits across every `meta.json`, `description*.md` and `notes*.md` under `problems`. Order 861 (`au-torchaudio-loudness-gate-ledger`) already covers `biquad`, `lfilter`, `_lfilter`, `_lfilter_core_generic_loop` and the two-stage K-weighting IIR recursion, judging ITU-R BS.1770-4 loudness gating. Order 869 therefore does NOT re-judge the IIR recursion: `lfilter` is replaced by an explicit input spy that records only the six coefficients plus `clamp` / `batching` / `dtype` handed over by `biquad`, the same technique as the `spectrogram` spy in order 868.

### Official source

```text
pytorch/audio main::src/torchaudio/functional/filtering.py  (1702 lines)
  _dB2Linear(x)                       lines 12-13
  allpass_biquad(...)                 lines 70-105
  band_biquad(...)                    lines 108-158
  bandpass_biquad(...)                lines 161-204
  bandreject_biquad(...)              lines 207-241
  bass_biquad(...)                    lines 244-292
  biquad(...)                         lines 295-333
  deemph_biquad(...)                  lines 417-462
  equalizer_biquad(...)               lines 630-669
  lowpass_biquad(...)                 lines 1102-1132
  riaa_biquad(...)                    lines 1299-1360

  b0 = ...; b1 = ...; b2 = ...; a0 = ...; a1 = ...; a2 = ...
  return biquad(waveform, b0, b1, b2, a0, a1, a2)

  output_waveform = lfilter(waveform,
      torch.cat([a0, a1, a2]),   # a_coeffs FIRST
      torch.cat([b0, b1, b2]))  # b_coeffs SECOND
```

`official869.py` extracts those 11 top-level functions with `ast.get_source_segment` and execs them into one shared namespace; NO official body is modified. The only injected objects are real `torch`, `math`, `warnings`, `typing`, `F`, `_IS_TORCHAUDIO_EXT_AVAILABLE=False` and one `lfilter` spy. The global default dtype is pinned to `float64`. Measurements show `math.sin/cos` differ from `torch.sin/cos` by at most 1.1e-16 inside the input domain and `exp/sqrt` agree exactly, so the pure-Python reference can use `math` for bit-for-bit cross-checks.

## 2. Conventions

### Eight modes

`solve(mode, **kw)` dispatches eight ways: `coeffs` returns the full ledger `{kind, a, b, a_norm, b_norm, a0_raw, waveform_shape, output_shape, params}`; `call` returns the spy record `waveform_shape` / `a_coeffs` / `b_coeffs` / `clamp=True` / `batching=True` / `dtype='torch.float64'`; `db2linear` returns `exp(x*ln10/20)`; `presets` returns two deemph plus four riaa rows; `errors` returns seven boundary cases; `census` returns 14 source facts; `grid` returns all rows for nine kinds at their legal sample rates; `biquad_order` calls `biquad` directly with explicit `b0..a2` to verify coefficient ordering.

### Non-finite values and input domain

torch tensor division, sqrt and trigonometric functions yield `nan` / `+/-inf` at boundaries instead of RAISING (Python `math` would raise). The reference must emulate this IEEE behaviour explicitly. To keep fixtures valid Python literals, every non-finite result is canonicalized to the string `"nan"`, `"inf"` or `"-inf"`: `Q=0` gives `a=[inf, ..., -inf]`, `sample_rate=0` gives all `nan`, a negative sample rate flips inf signs, and `band` with `Q=0` gives `b0="nan"`.
`deemph_biquad` accepts only 44100 / 48000, and `riaa_biquad` only 44100 / 48000 / 88200 / 96000; any other rate raises `ValueError` with the verbatim official message. The `grid` mode walks only legal rates for those two kinds and 8000 / 16000 / 44100 for the other seven. A ragged nested `waveform` reproduces torch.tensor's dimension message `expected sequence of length N at dim 1 (got M)`.

## 3. Formulas and quirks of the nine kinds

```text
w0    = 2*pi*freq / sample_rate
alpha = sin(w0) / 2 / Q          (RBJ cookbook)
A     = exp(gain / 40 * ln(10)) = _dB2Linear(gain/2)

lowpass     b = [(1-cos w0)/2, 1-cos w0, (1-cos w0)/2]
            a = [1+alpha, -2cos w0, 1-alpha]
allpass     b = [1-alpha, -2cos w0, 1+alpha]  == reverse(a)
bandreject  b = [1, -2cos w0, 1]
bandpass    temp = sin(w0)/2 if const_skirt_gain else alpha
            b = [temp, 0, -temp]  (skirt ratio == Q)
equalizer   b0/b2 = 1 +/- alpha*A ; a0/a2 = 1 +/- alpha/A
bass        SoX temp1/temp2/temp3 ; ALL SIX coefficients
            divided by a0 BEFORE calling biquad  (a0_raw == 1.0)
band        bw_Hz = freq/Q ; a2 = exp(-2*pi*bw_Hz/sr)
            a1 = -4*a2/(1+a2)*cos(w0) ; b1 = b2 = 0
            b0 = sqrt(1 - a1^2/(4*a2))*(1-a2)
            noise=True -> b0 = sqrt(((1+a2)^2-a1^2)*(1-a2)/(1+a2))
deemph      fixed (central_freq, width_slope, gain) per rate
            alpha uses sqrt((A+1/A)*(1/width-1)+2)
riaa        zeros/poles polynomials per rate, then g-scaled so
            |H(exp(j*2*pi*1000/sr))| == 1
```

(1) `bass` is the ONLY kind that pre-normalizes: it computes six raw coefficients and divides ALL of them by `a0` before calling `biquad`, so `a0_raw` is always `1.0` and `a_norm == a`; the other eight kinds pass unnormalized coefficients. (2) `equalizer` is asymmetric between `alpha*A` and `alpha/A`, so `b != a` (identity holds only at `gain=0` where `A=1`); `bass` likewise degenerates to an identity filter at `gain=0`.
(3) `band` uses a DIFFERENT SoX formulation: `a0=1`, `a2=exp(-2*pi*bw_Hz/sr)`, `a1=-4*a2/(1+a2)*cos(w0)`, `b1=b2=0`, with an extra `mult` factor when `noise=True`; the discriminant `1 - a1^2/(4*a2)` is provably non-negative in the input domain. (4) `deemph` and `riaa` use plain Python `math` (the rest use torch tensors) and their fixed parameter tables are valid only at the designated sample rates; `riaa` additionally computes `g` from the real/imaginary parts of the 1 kHz response for 0 dB normalization.
(5) `biquad` puts a FIRST and b SECOND: `lfilter(waveform, a_coeffs, b_coeffs, ...)` receives `a_coeffs=[a0,a1,a2]` and `b_coeffs=[b0,b1,b2]`. (6) `Q<0` or `sample_rate<0` flips the sign of `alpha`, swapping `a0` and `a2` for `lowpass`; a `cutoff_freq` above Nyquist raises nothing, `cos(w0)` simply keeps following the formula.

## 4. Judge ledger

`fixtures869.py` exports 132 constants from the verbatim official environment, and `cases869_checks.py` supplies 53 `check_*(solve)` functions: default coefficients and three-tier parameter grids for all nine kinds, four gain levels each for `bass` / `equalizer`, the `band` noise branch and negative-discriminant nan, the `bandpass` skirt ratio, all legal `deemph` / `riaa` presets and illegal-rate messages, the six-rate `grid`, seven `errors` cases, a-b ordering and non-finite canonicalization in `call`, ragged-waveform messages, mixed-sign inf, the 14 `census` source facts plus the fresh-dict guarantee, and the complete `unknown mode` message.
The three-layer independent oracle (`verify869.py`) shares NO algorithm with the reference: trig values come from `cmath.exp(1j*x)`, `A` from `10**(gain/40)`, `exp` from `cmath.exp`, `sqrt` from `x**0.5` and pi from `4*atan(1)`; the RIAA zero/pole polynomials are rebuilt with exact `Fraction` rationals; z-domain transfer-function properties (all-pass magnitude 1, unity DC gain, notch, peaking gain, 1 kHz normalization) then cross-pin the result.

## 5. Implementation notes

The reference uses only `math`. The `_div` / `_sin` / `_cos` / `_sqrt` / `_exp` helpers emulate torch's IEEE semantics so Python never raises on division by zero or sqrt of a negative; `_clean` maps non-finite values to stable strings; `_shape_of` reproduces `torch.tensor` ragged detection; `_coeffs` reports both raw a/b and the `a0`-normalized a_norm/b_norm. The nine kinds are implemented independently without shared intermediates, so each can be judged in isolation.
In-domain float comparisons use `rtol=1e-8`; `census` and error messages compare exactly as strings. Apart from `unknown mode` raising `ValueError`, all exceptions inside `errors` mode are folded into `{type, msg}` dicts whose text matches the official wording verbatim.
