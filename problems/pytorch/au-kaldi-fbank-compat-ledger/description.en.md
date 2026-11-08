# Kaldi-compatible frontend ledger: the 1127/700 mel scale, VTLN three-piece linear warp with four overlapping assignments, the warp branches and (num_bins,1) shape quirk of get_mel_banks, snip_edges reflect padding of _get_strided, the preprocessing chain order and raw_energy, and the switch semantics of fbank/mfcc with htk_compat tail orchestration

## 1. Background

torchaudio's `compliance/kaldi.py` is a bit-compatible implementation of Kaldi fbank / mfcc features. This problem decomposes the whole frontend pipeline into five independently checkable numerical ledgers, all implemented with plain Python floats, with dither noise always passed in explicitly so that no randomness source is involved. Unlike the basic component problems at orders 340-343, this problem pins down the parameter semantics and official quirks of the compatibility layer rather than the definition of a single window function or filterbank.

### Official source locations

- `src/torchaudio/compliance/kaldi.py`: `_feature_window_function` / `_get_strided` / `_get_window` / `_get_log_energy` / `get_mel_banks` / `vtln_warp_freq` / `vtln_warp_mel_freq` / `spectrogram` / `fbank` / `mfcc`
- `src/torchaudio/functional/functional.py::create_dct`: the source of the DCT matrix for mfcc

The judging environment has no `torchaudio`, so a literal official execution environment was built by injecting a fake `torchaudio.functional.create_dct`; the probes run the real official implementation rather than a rewrite.

## 2. Conventions

### Inputs and outputs

A waveform is a one-dimensional list `[n]`, framed data is `[m][padded_window_size]`, a filterbank is `[num_bins][num_fft_bins]`, and features are `[m][d]`. Each mode returns a dictionary whose keys mirror the official variable names. Every invalid input raises `ValueError` with a message that names the violated constraint.

### Numerical precision and the float32 quirk

All values are Python floats (float64). `EPS` is always `torch.finfo(torch.float).eps = 1.1920928955078125e-07`: the official `EPSILON` is a float32 scalar and `_get_epsilon` only calls `.to(dtype)`, so its value does not change with dtype.

Inside the official `get_mel_banks`, `torch.arange(int64) * float` is demoted to float32, so even with float64 input the `center_freqs` at a few thousand Hz carry about `1e-3` of quantization error, which accumulates through the DCT into roughly `2.5e-5` absolute error on mfcc `features`. official-vs-reference therefore uses `tol=1e-6, rel=1e-5` by default and `tol=5e-5, rel=0.0` for mfcc `features` alone; oracle-vs-reference always uses `1e-7`.

## 3. Mode 1: vtln_warp

### Three-piece linear warp and four overlapping assignments

`vtln_warp_freq` first computes two inflection points and the left/right slopes, then applies four mutually overlapping assignments over the whole frequency range:

```text
l = vtln_low * max(1.0, vtln_warp)
h = vtln_high * min(1.0, vtln_warp)
scale = 1.0 / vtln_warp
Fl = scale * l
Fh = scale * h
scale_left = (Fl - low_freq) / (l - low_freq)
scale_right = (high_freq - Fh) / (high_freq - h)
```

The official source comment says `order of operations matter here (since there is overlapping frequency regions)`. The four assignments are, in order: (1) `after_h` (`freq >= h`) writes `high_freq + scale_right*(freq-high_freq)`; (2) `before_h` (`freq < h`) writes `scale*freq`; (3) `before_l` (`freq < l`) writes `low_freq + scale_left*(freq-low_freq)`, overwriting the low-frequency overlap of (2); (4) `outside_low_high_freq` (`freq < low_freq` or `freq > high_freq`) finally writes out-of-range frequencies back unchanged. The result satisfies `F(low_freq)=low_freq`, `F(high_freq)=high_freq`, and the middle segment `l <= f <= h` is `f/vtln_warp`. `vtln_warp_mel_freq` is `mel_scale(vtln_warp_freq(inverse_mel_scale(mel)))`; when warp is not 1, the left / center / right mel columns are all warped.

### Domain constraints

`vtln_low <= low_freq` raises `vtln_low must be higher than low_freq`; `vtln_high >= high_freq` raises `vtln_high must be lower than high_freq`; `vtln_warp <= 0` raises `vtln_warp must be positive`; out-of-range inflection points raise `inflection points out of range`.

## 4. Mode 2: mel_banks

### Mel scale and delta

The mel scale is `1127*ln(1+f/700)` and its inverse is `700*(exp(m/1127)-1)`, not HTK's `2595*log10(1+f/700)`.

```text
mel_low_freq = 1127.0 * ln(1 + low_freq / 700.0)
mel_high_freq = 1127.0 * ln(1 + high_freq / 700.0)
mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)
```

The denominator of `mel_freq_delta` is `num_bins + 1`. `left_mel` / `center_mel` / `right_mel` are `mel_low_freq + bi*delta` / `+(bi+1)*delta` / `+(bi+2)*delta`. `num_fft_bins = window_length_padded / 2`, `fft_bin_width = sample_freq / window_length_padded`, and the mel grid is `mel(fft_bin_width*k)`. `high_freq <= 0` adds the nyquist, and `vtln_high < 0` adds the nyquist.

### Two warp branches

```text
up = (mel[k] - lm) / (cm - lm)
down = (rm - mel[k]) / (rm - cm)
warp == 1.0 : max(0.0, min(up, down))
warp != 1.0 : lm < mel[k] <= cm -> up ; cm < mel[k] < rm -> down ; else 0.0
```

`vtln_warp == 1.0` takes `torch.max(zeros, torch.min(up_slope, down_slope))`; `vtln_warp != 1.0` takes the mask branch with strict / non-strict inequalities at the boundaries. The two branches produce different numbers and cannot replace each other.

### The center_freqs shape quirk

The official docstring claims `center_freqs` has shape `(num_bins,)`, but it actually returns `(num_bins, 1)`; this problem keeps the nested list. `padded_window_size` must be even and no smaller than `window_size`, and `num_bins` must be greater than 3.

## 5. Mode 3: framing

### Framing arithmetic

```text
snip_edges=True  : m = 1 + (num_samples - window_size) // window_shift
snip_edges=False : m = (num_samples + window_shift // 2) // window_shift
pad = window_size // 2 - window_shift // 2
```

With `snip_edges=True` and a waveform shorter than the window, zero frames are returned (the `(0,0)` semantics). With `snip_edges=False`, reflect padding uses `waveform[::-1]`: when `pad > 0` the concatenation is `reversed[-pad:] + waveform + reversed`, and when `pad <= 0` it is `waveform[-pad:] + reversed`. Reflect padding is neither zero padding nor `replicate`.

### Preprocessing chain order

```text
dither -> remove_dc_offset -> raw_energy -> preemphasis -> window
       -> right zero pad -> post_energy (if not raw_energy)
```

- `dither`: `noise[i][j] * dither`, with the noise passed in explicitly.
- `remove_dc_offset`: subtract each frame's mean (divide by `window_size`).
- `raw_energy=True` computes energy **before** preemphasis and windowing; `False` computes it **after** windowing and zero padding.
- `preemphasis`: the official `F.pad(..., (1,0), mode='replicate')[:, :-1]` takes `x[0]` as the first sample, so `j=0` is `(1-coeff)*x0`.
- `_get_log_energy`: `log(max(sum(x^2), EPS))`; `energy_floor` is applied as a max **in the log domain**, `max(le, log(energy_floor))`; `energy_floor=0` returns `le` directly.
- Five window branches: `hamming` / `hanning` / `povey` / `rectangular` / `blackman`; `povey = hann^0.85`; blackman is the official hand-written formula, which equals `torch.blackman_window` exactly at the default `blackman_coeff=0.42` but not at `0.5`; an unknown window raises `Invalid window type`.

### Default values

```text
snip_edges=True, window_type=povey, blackman_coeff=0.42, dither=0.0
remove_dc_offset=True, preemphasis_coefficient=0.97
raw_energy=True, energy_floor=1.0
```

With an all-zero waveform, the default `energy_floor=1.0` must clamp log-energy to `log(1)=0`; the `povey` window does not depend on `blackman_coeff`, so verifying the default coefficient must explicitly take the blackman branch.

## 6. Mode 4: fbank

### Spectrum and mel energies

Each framed row from `_get_window` goes through `torch.fft.rfft(x).abs()` for `k=0..N/2`, squared when `use_power` to give `spectrum`; `bins` get one right zero column to become `(num_bins, num_fft_bins+1)` and are dotted with the spectrum to give `mel_energies`; `use_log_fbank` applies `log(max(v, EPS))`. `use_energy` concatenates `signal_log_energy`, and `subtract_mean` subtracts per-column means as CMS.

### Switch semantics

```text
use_power      : spectrum = |X|^2   else |X|
use_log_fbank  : log(max(E, EPS))   else E
use_energy     : htk_compat -> [E..., sig] else [sig, E...]
subtract_mean  : per-column mean subtraction over frames
```

The official `spectrogram` has its column 0 overwritten by `signal_log_energy`, not `log(|X0|^2)`; this problem returns the energy separately as the `signal_log_energy` field.

## 7. Mode 5: mfcc

### DCT matrix and lifter

```text
dct = create_dct(num_mel_bins, num_mel_bins, "ortho")
dct[:, 0] = sqrt(1 / num_mel_bins)
dct = dct[:, :num_ceps]
lifter[i] = 1 + 0.5 * L * sin(pi * i / L)
```

mfcc forces `use_power=True` / `use_log_fbank=True` / `subtract_mean=False` when calling fbank. `_get_dct_matrix` transposes `create_dct(..., 'ortho')` and force-overwrites the first column with `sqrt(1/n)` (overriding the ortho `1/sqrt(2)` factor). C0 of the lifter is always `1.0`, and the official skips the lifter when `cepstral_lifter=0.0` (this problem skips it too, to avoid division by zero). `num_ceps > num_mel_bins` raises `num_ceps cannot be larger than num_mel_bins`.

### htk_compat tail orchestration

```text
use_energy: sig = row[num_mel_bins if htk_compat else 0]
            mel_offset = 0 if htk_compat else 1
            C0 = sig (overwrite)
htk_compat: energy = row[0] (or row[0] * sqrt(2.0) if not use_energy)
            out = row[1:] + [energy]
```

With `use_energy`, the energy column is first extracted from the fbank features (last under `htk_compat`, first otherwise); after the DCT, C0 is overwritten with that energy; `htk_compat` then moves C0/energy to the tail. `subtract_mean` is applied to the mfcc output only at the very end.

## 8. Official quirk list

1. `center_freqs` is actually `(num_bins, 1)` while the docstring says `(num_bins,)`.
2. The four VTLN assignments rely on overlapping regions, so their order cannot be swapped; out-of-range frequencies are kept unchanged.
3. int64 times float inside `get_mel_banks` is demoted to float32, giving `center_freqs` about `1e-3` of quantization error.
4. Column 0 of `spectrogram` is overwritten by `signal_log_energy`, not `log(|X0|^2)`.
5. At the default `blackman_coeff=0.42`, the official hand-written formula equals `torch.blackman_window` exactly.
6. `energy_floor` is a max in the log domain, not a clamp in the energy domain.
7. `snip_edges=False` uses reflect padding rather than zero padding.
8. The mel scale is the `1127/700` natural-log convention, not HTK's `2595/log10`.
9. With `cepstral_lifter=0.0` the lifter is skipped and C0 is always `1.0`.
10. When warp is not 1, left / center / right mel are all warped, changing both bins and center_freqs.

## 9. Common mistakes

1. Using HTK's `2595*log10(1+f/700)` for mel.
2. Dividing `mel_freq_delta` by `num_bins` instead of `num_bins + 1`.
3. Flattening `center_freqs` to `(num_bins,)`.
4. Swapping the order of the four VTLN assignments or dropping the final `outside_low_high_freq` overwrite.
5. Using `min` for `l` / `max` for `h`, or writing `scale` as `vtln_warp`.
6. Still using the `max/min` branch when `warp != 1.0`, or missing the strict inequalities of the mask boundaries.
7. Warping only the center mel and forgetting left / right when `warp != 1.0`.
8. Using zero padding instead of reflect padding for `snip_edges=False`.
9. Not returning zero frames when `snip_edges=True` and the waveform is too short.
10. Writing `pad` with float division instead of `//`.
11. Wrong preprocessing chain order: computing `raw_energy` after windowing.
12. Taking `x[-1]` or `0` as the first preemphasis sample instead of the replicate `x[0]`.
13. Applying `energy_floor` in the energy domain instead of the log domain.
14. Forgetting `**0.85` for `povey`.
15. Wrong sign or angle in the hand-written blackman formula.
16. Dropping the `dither` multiplier or not using the explicit `noise`.
17. Forgetting the right zero column of `bins`.
18. Putting the energy column at the wrong head/tail position under `htk_compat` with `use_energy`.
19. Forgetting to force the dct first column to `sqrt(1/n)`.
20. Still computing the lifter when `cepstral_lifter=0.0`, causing division by zero.
21. `htk_compat` not moving C0/energy to the tail.
22. Changing the default `energy_floor=1.0` / `blackman_coeff=0.42`.
23. Not raising `unknown mode` for an unrecognized mode.
24. Asserting only that "an exception is raised" for domain constraints without checking the message keyword.

## 10. Self-check

- With an all-zero waveform and the default `energy_floor=1.0`, `signal_log_energy[0] == log(1.0) == 0`.
- mel and inverse mel are mutual inverses on positive frequencies.
- `bins` / `center_freqs` must differ between `warp=1` and `warp!=1`.
- `num_ceps <= num_mel_bins` and `lifter_coeffs[0] == 1.0`.
- The three-layer cross-check, test cases, and mutation must all pass before landing.
