# 864 Problem notes: torchaudio speed / pitch-shift orchestration ledger

## 1. Selection and duplicate check

Order 864, collection `audio` 15->16, slug `au-torchaudio-speed-pitch-ledger`. Sources are `speed` (2385-2423), `pitch_shift` (1596-1641), `_stretch_waveform` (1644-1693) and `_fix_waveform_shape` (1696-1719) in `functional.py`. Library-wide searches for `pitch_shift`, `_stretch_waveform`, `_fix_waveform_shape`, `speed`, `bins_per_octave`, `phase_advance` and `len_stretch` in the meta / description / notes of existing problems return no hits; `pitch_shift` is only a candidate mention in the 860 notes. The function set is fully disjoint from 857 `phase_vocoder`, 859 sinc kernel, 860 `torch.istft` / Griffin-Lim, 862 MVDR and 863 FID.

## 2. Official source and verbatim environment

`official864.py` extracts the four top-level definitions with `ast.get_source_segment` and execs them into a shared namespace; no official body is modified. `torch` and `math` are forwarding proxies (`stft` / `istft` / `gcd` also log their arguments) and `phase_vocoder` / `resample` are recording wrappers returning explicit caller-supplied tensors. The global default dtype is pinned to `float64` so `linspace` is comparable with the pure-Python reference. `ledger()` reads rate / hop / win / window_len / freq_bins / pack_shape / len_stretch / resample rates / raw gcd inputs / out_lengths from the call log and is the single authoritative source for probes and fixtures.

## 3. Mechanism ledger

`rate = 2**(-n_steps/bins)` with `int(sample_rate/rate)` truncation; `hop=n_fft//4` / `win=n_fft` / default Hann window; `phase_advance = linspace(0, pi*hop, spec_f.shape[-2])` sized by the FREQUENCY bin count; `len_stretch = int(round(ori_len/rate))` with ties-to-even; `reshape(-1, ori_len)` packing; `_fix_waveform_shape` trim / trailing zero-pad plus `view(shape[:-1]+[ori_len])`; `speed` performs two `int()` truncations, reduces with `math.gcd` before resample, and applies `ceil(lengths*tgt/src).to(dtype)`.

## 4. Three-layer verification and mutation

Twelve probe segments give `427/427 ALL GREEN` (P1 verbatim source comparison, P2 rate, P3 int truncation, P4 defaults, P5 linspace, P6 packing, P7 banker's rounding, P8 stft/istft kwargs, P9 trim/pad/exact, P10 gcd, P11 ceil+dtype, P12 end-to-end census). The three-layer cross-check reports `13326 checks / 52 configs / 0 fails`: the official side runs the real verbatim environment, the reference is a pure-Python orchestration, and the independent oracle uses `exp+log`, `n_fft>>2`, `Fraction` steps, `Decimal ROUND_HALF_EVEN`, a hand-written Euclid loop, `Decimal ROUND_CEILING` and index-driven chunk reshaping; no algorithm is shared. The judge driver reports `152/152 ALL ACCEPTED`.

Round one: 160 mutants / 135 CAUGHT / 25 SURVIVED / 0 ERROR. Thirteen hardening cases (MODES order in the unknown-mode message, an empty waveform, default dtype must stay float, a two-distinct-row reshape, 1-D padding, a single-frequency-bin linspace, a 3-D spec_f, 3-D default injection through `_nest`, a custom `fix_shape` target length, `pitch_shift` independence from `fix_*`, and the injected-only `stretched` value) reduced this to one survivor, and `d-nest-reverse` was then archived with a proof based on replicating the same row across all leading dims followed by `_fix` flatten-and-reshape. Final result: **160 mutants, 13 documented equivalents excluded, 147/147 CAUGHT, 0 SURVIVED, 0 ERROR**. The equivalents are `v-shape-descend`, `v-pack-conv`, `v-pack-no-conv`, `v-neednum-cast`, `v-needint-cast`, `v-torchsize-join`, `n-linspace-zero`, `n-fix-trim-gt`, `n-fix-pad-lt`, `d-stretched`, `d-shifted-len`, `d-nest-reverse` and `l-lengths-reshape`; each proof is written in the `mut864.py::EQUIVALENT` comment.

## 5. Scope cuts and engineering lessons

The judged domain excludes STFT/ISTFT numerics, `phase_vocoder` phase accumulation, the sinc kernel, window coefficients, audio IO and model forwards; waveforms, spectra and windows are explicit inputs, pure Python with no numpy/torch/scipy and no global RNG. The biggest trap is that the DEFAULT injected waveform has identical rows: this lets mutants such as 'ignore idx while reshaping' and 'reverse the leading-dimension order' survive, so a dedicated two-distinct-row case is mandatory. The second trap is that the raw gcd inputs of `speed` can only be recovered from the logged `math.gcd` arguments (the official body passes the REDUCED rates to resample), so the proxy must log `gcd` and not just `resample`. The third trap is that dimension `-2` of `spec_f` is the frequency-bin count: using the frame count or hard-coding `n_fft//2+1` is exposed by injecting a spectrum with a different row count.

## 6. Official quirks and duplicate-check conclusion

Same as section 5 of `description.md`: `int(sample_rate/rate)` gives 20158 on 20158.738; the linspace length is the frequency-bin count; ties-to-even sends 4.5 to 4 and 7.5 to 8; `0.7*44100=30869.999...` truncates to 30869 so the gcd collapses to 1; resample receives the gcd-reduced rates; `_fix_waveform_shape` normalises the last axis before the view restores leading dims; and `stretched` is validated but never consumed by the orchestration layer. Additionally `n_steps` passes through `float()` and allows fractional steps, `bins_per_octave=0` raises `ZeroDivisionError`, and an explicit `window` length is never validated. Library-wide duplicate checks confirm the function set is fully disjoint from orders 855 and 857-863.

## 7. Registration and guards

`register864.py` registers with minimal intrusion, growing `audio` from 15 to 16 and appending the 864 convention paragraph to both descriptions; verification expects `842 registered = 842 disk / max_order=864 / no dangling / no unregistered`. `todos.md` is updated to 842 problems / audio 16 / next batch from 865. The library-wide bilingual check expects `0 error / 15 warning` (all 15 historical); the short guard uses an isolated `--basetemp`, and the full guard result is backfilled after its background run. Per the standing convention, no git commit is made until the user confirms.
