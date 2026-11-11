# 860 authoring notes: torchaudio Griffin-Lim phase-recovery ledger

## 1. Topic selection and deduplication

Order 860, collection `audio` 11->12, slug `au-torchaudio-griffinlim-phase-ledger`. It takes four functions from lines 148-353 of `functional.py`: `inverse_spectrogram`, `_get_spec_norms`, `_get_complex_dtype` and `griffinlim`. Library-wide searches for `griffinlim`, `inverse_spectrogram`, `loudness`, `pitch_shift`, `spectral_centroid`, `mvdr`, `beamform`, `edit_distance` and `frechet` returned no hits. `mask_along_axis` is covered by `asr-specaugment-mask` (order 693) and is avoided. The nine MVDR functions need a pure-Python Hermitian eigendecomposition and are kept as a higher-risk backup; `loudness`, `pitch_shift` and `_stretch_waveform` overlap partially with the `phase_vocoder` work in 857.
It shares `functional.py` with 855, 857, 858 and 859 but uses disjoint functions, and is separated from 344 polyphase resampling by algorithm family.

## 2. Verbatim official environment

`.workbuddy/scratch/torchaudio860/official860.py` AST-extracts and `exec`s the four functions, injecting `math/typing/warnings/torch/Tensor/Optional/Union/_CPU`, and wraps them with `comp_norms`, `comp_complex_dtype`, `comp_griffinlim`, `comp_inverse`, `raw_istft`, `raw_stft`, `rand_angles` and `window_l2`. ★ Key engineering decision: `rand_init=True` is not stripped; the driver uses `angles_override` to temporarily replace `torch.rand` with a fixed complex128 matrix, turning the random initial phase into an explicit argument and matching the library-wide convention of explicit RNG inputs.
Two feasibility constraints pinned during source review: the `torch.istft` OLA denominator requires that the window is not zero at both ends (a symmetric Hann with an extended `length` immediately raises `window overlap add min`), and the `rebuilt` frame count inside the `griffinlim` loop is `1 + length//hop`, which must equal the input `specgram` frame count.

## 3. Twelve probe segments

```text
P1  verbatim source / P2  _get_spec_norms / P3  _get_complex_dtype
P4  momentum guard + n_iter=0 / P5  istft OLA, center, length
P6  reflect-pad framed DFT / P7  win_length < n_fft zero padding
P8  Hermitian mirror / DC-Nyquist imaginary
P9  normalized + window L2 / P10 inverse_spectrogram quirks
P11 rand_init / explicit angles / P12 griffinlim end-to-end
PROBE 860 DONE ok=105 fail=0
```

The two first-round red assertions were wrong probe expectations: a 16-sample padded waveform at `hop=2` gives 5 frames, not 7; and `n_iter=0` still uses the `rand_init` initial phase (the official code initialises `angles` before entering the loop), so the two `rand_init` settings differ. Both were rewritten to the measured official behaviour and became two documented quirks.

## 4. Three-way verification

`verify860.py` compares the verbatim official environment, the reference and an independent oracle over nine parameter groups (`n_fft=8/10/16`, `hop=2/3/4/5/8`, rectangular / periodic Hann / `win_length=6` windows, both `center` settings, both `normalized` settings and several `length` values). The oracle deliberately avoids `cmath`: it expands the Hermitian inverse DFT into real cosine/sine pairs and gathers frames per output sample instead of scattering. The first round was 2880 green checks but skipped 16 case-B GL cases because reflect padding needed a longer signal; raising the GL frame count to `T = n_fft//2//hop + 3` produced **2960 checks ALL GREEN / 0 SKIP**.

## 5. Judge cases

`test_cases_staged.py` holds 45 top-level `check_*(solve)` cases whose numbers come from the official value snapshots in `fixtures860.txt`; `driver860.py` and both selftest modes report `45/45 ALL ACCEPTED`. With the first 32 cases there were 7 red results: five came from `**GEOM` expansion conflicting with explicit overrides (fixed with a `_g(**overrides)` factory), and two came from expecting `envelope_min == 1.0` when a rectangular window at `hop=4` has a constant envelope of 2.0 across the output region.

## 6. Mutation testing

```text
round 1: 134 CAUGHT / 92 SURVIVED / 0 ERROR
fix    : _near() must assert, not return a bool
round 2: 178 CAUGHT / 48 SURVIVED / 0 ERROR
patch  : +11 hardening checks, 4 equivalence rulings
round 3: 220 CAUGHT / 2 SURVIVED / 0 ERROR
patch  : +2 hardening checks (n_fft=16 default, zero phase)
round 4: 222 CAUGHT / 0 SURVIVED / 0 ERROR
226 total / 4 equivalent / 222 active -> ALL MUTANTS CAUGHT
```

★ The 92 first-round SURVIVED mutants exposed a bug in the judge helper: `_near` returned a bool instead of asserting internally, so every bare `_near(...)` call was a no-op. Restoring the historical convention (recursive `assert`) immediately cut the list to 48. Those 48 came from untested defaults, error checks matching only keywords instead of full messages, untested bool / string / non-numeric branches, unasserted metadata fields, and missing-argument branches; 11 hardening groups reduced them to 2.
The final two: `g-win-default` needs `n_fft=16` with `win_length` omitted to reveal the difference between a literal 8 and `n_fft`; `l-norm-noeps` needs an iteration whose rebuilt spectrum is exactly zero, achieved with an all-zero `angles_init` that makes `spec*angles` and therefore the rebuilt spectrum zero, where the official `+1e-16` avoids `0/0` and yields an all-zero waveform. Both hardening expectations were taken from the verbatim official environment.
Four true equivalents: `require_float` is never called with a `low` bound, so that branch and its message are unreachable; with `win_length == n_fft` the general zero-padding path equals the direct return; and `buf[i] if i < total else 0.0` is literally equivalent to `0.0 if i >= total else buf[i]`.

## 7. Stripped conventions

(1) The judged domain takes only two-dimensional `(freq, frames)` spectrograms; the official batch pack/unpack is excluded. (2) The `rand_init` RNG becomes an explicit `angles_init`, with the official `torch.rand(complex128)` values supplied as fixed-seed snapshots. (3) Complex matrices are expressed as two real matrices `real` / `imag`, while `angles_init` uses `[re, im]` pairs. (4) `pad_mode` is recorded as present in the signature but never used, not simulated. (5) Numerics are float64 throughout; `float32` / `float16` appear only in the `complex_dtype` mapping table. (6) `torch.fft` is rewritten as an O(N^2) complex-exponential DFT, cross-checked by a trigonometric independent oracle.

## 8. Official quirk list

Same as section 8 of `description.md`: reflect padding excluding the boundary, left-biased zero padding, Hermitian imaginary discarding, envelope division and the NOLA threshold, length zero padding, opposite `normalized` directions, the True/string equivalence in norms, pad and pad_mode, the momentum transform, in-place tprev with its unnormalised copy, +1e-16, `n_iter=0` still using the initial phase, and the frame-count constraint linking length and hop.

## 9. Deduplication and landing

Before landing, `max_order=859` was confirmed and searches for `griffinlim` / `inverse_spectrogram` returned no hits library-wide. `collections.json` is registered with minimal intrusion and `todos.md` is updated to 838 problems / audio 12 / next batch from 861. `land860.py` lands 11 files and verifies the three core artifacts against `variants/pytorch/` with `filecmp`.
