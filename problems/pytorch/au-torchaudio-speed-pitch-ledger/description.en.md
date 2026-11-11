# torchaudio speed / pitch-shift orchestration ledger: the rate exponent and int() truncation, hop/win/window default resolution, the phase_advance linspace sized by the FREQUENCY bin count, banker's rounding of len_stretch, trim/pad plus view in _fix_waveform_shape, and gcd reduction with ceil+cast in speed

## 1. Background

`pitch_shift`, `_stretch_waveform`, `_fix_waveform_shape` and `speed` in `torchaudio.functional` form a pure orchestration layer: they perform no numerical DSP themselves, only resolve parameter defaults, compute rates and lengths, reduce sample-rate ratios, and repair the output shape. All heavy work is delegated to `torch.stft` / `torch.istft` / `phase_vocoder` / `resample`. This problem reproduces that orchestration ledger in pure Python and splits it into ten independently judged modes.
Library-wide searches for `pitch_shift`, `_stretch_waveform`, `_fix_waveform_shape` and `speed` in the meta, description and notes of existing problems return no hits (`pitch_shift` appears only as a candidate mention in the 860 notes). Order 857 covers `phase_vocoder` itself, 859 the sinc resampling kernel, 860 `torch.istft` / Griffin-Lim, and 862 / 863 cover MVDR and FID; the function sets are disjoint.

### Official source

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  speed                    (line 2385, 39 lines)
  pitch_shift              (line 1596, 46 lines)
  _stretch_waveform        (line 1644, 50 lines)
  _fix_waveform_shape      (line 1696, 24 lines)

  pitch_shift:
    waveform_stretch = _stretch_waveform(...)
    rate = 2.0 ** (-float(n_steps) / bins_per_octave)
    waveform_shift = resample(waveform_stretch, int(sample_rate / rate), sample_rate)
    return _fix_waveform_shape(waveform_shift, waveform.size())

  _stretch_waveform:
    hop = n_fft // 4 ; win = n_fft ; hann_window(win) if window is None
    waveform = waveform.reshape(-1, shape[-1])
    spec_f = torch.stft(..., center=True, pad_mode='reflect',
             normalized=False, onesided=True, return_complex=True)
    phase_advance = torch.linspace(0, pi * hop, spec_f.shape[-2])[..., None]
    spec_stretch = phase_vocoder(spec_f, rate, phase_advance)
    len_stretch = int(round(ori_len / rate))
    return torch.istft(spec_stretch, ..., length=len_stretch)

  _fix_waveform_shape:
    trim to ori_len, else zero-pad, then view(shape[:-1] + [ori_len])

  speed:
    src = int(factor * orig_freq) ; tgt = int(orig_freq)
    gcd = math.gcd(src, tgt) ; src, tgt = src // gcd, tgt // gcd
    out_lengths = ceil(lengths * tgt / src).to(lengths.dtype)
    return resample(waveform, src, tgt), out_lengths
```

### Verbatim environment and injection scope

`official864.py` extracts the four definitions from the pinned 2511-line upstream source with `ast.get_source_segment` and execs them into one shared namespace; no official body is modified. The four heavy DSP dependencies are injected as RECORDING wrappers: `torch` and `math` are forwarding proxies that additionally log the call arguments of `stft` / `istft` / `gcd`, while `phase_vocoder` and `resample` are recording wrappers returning caller-supplied tensors. The rate exponent, `int()` truncation, gcd reduction, `round()` ties-to-even, linspace length, default resolution, pack/view, ceil+cast and trim/pad therefore all really execute, and the logged arguments are the authoritative official ledger.
The cut is deliberate: reflect padding of STFT/ISTFT, polar phase accumulation of the vocoder, and windowed polyphase convolution of the sinc kernel are already covered verbatim by orders 860 / 857 / 859. With those values supplied explicitly, the judged domain contains no FFT, no randomness and no audio IO, so a pure-Python reference can be compared with the official ledger bit by bit in milliseconds.

## 2. Conventions

### Inputs

- Every argument is a keyword of `solve(mode, **kw)` and has a default. `waveform`, `spec_f`, `stretched`, `shifted`, `fix_waveform`, `speed_waveform`, `speed_out`, `window` and `speed_lengths` are explicit nested numeric lists (int / float; bool is rejected).
- Core scalars: `n_steps=4`, `bins_per_octave=12`, `n_fft=16`, `sample_rate=16000`, `speed_factor=1.5`, `speed_orig_freq=16000`; `win_length` / `hop_length` / `window` / `speed_lengths` default to `None`.
- `spec_f` has at least two dimensions; the official code takes `spec_f.shape[-2]` as the frequency-bin count, independent of the frame count (`shape[-1]`). The default injected spectrum is a zero `(n_fft//2+1) x 3` array.
- `fix_target_shape` is a target shape list; `speed_lengths_dtype` is one of `float64` / `float32` / `int64` / `int32` and decides whether the post-`.to(dtype)` ceil result is int or float.

### Ten modes

1. `rate` -> `2.0 ** (-n_steps / bins_per_octave)`, the rate shared by `pitch_shift` and `_stretch_waveform`.
2. `pitch_rates` -> `[int(sample_rate / rate), sample_rate]`; the first entry is TRUNCATED, not rounded.
3. `stretch_plan` -> `[hop_length, win_length, window_len, freq_bins]`, the four resolved defaults.
4. `phase_advance` -> the full `linspace(0, pi*hop_length, freq_bins)` list.
5. `len_stretch` -> `int(round(ori_len / rate))`, Python ties-to-even.
6. `pack_shape` -> `[prod(lead), ori_len]` after `waveform.reshape(-1, ori_len)`.
7. `fix_shape` -> the standalone `_fix_waveform_shape` output (configurable via `fix_waveform` / `fix_target_shape`).
8. `pitch_shift` -> the end-to-end output, equal to applying `_fix_waveform_shape` to the injected `shifted` waveform.
9. `speed_rates` -> `[int(factor*orig_freq), int(orig_freq), gcd, src//gcd, tgt//gcd]`.
10. `speed` -> `[speed_out, out_lengths]`, where `out_lengths` is `None` or the `ceil(lengths*tgt/src)` list.

## 3. Official mechanics, section by section

### 3.1 Rate and int() truncation

`rate = 2.0 ** (-float(n_steps) / bins_per_octave)`: `n_steps` is explicitly cast with `float()`, so a fractional step such as `5.5` is legal (the annotation says int but the body never checks). `bins_per_octave` need not be 12; 36 means 36 steps per octave. `pitch_shift` then uses `int(sample_rate / rate)` as the resampling source rate: with `n_steps=4, sample_rate=16000` the quotient is `16000/0.7937005... = 20158.738...`, and the official value is **20158** (`round` would give 20159, as would `ceil`).

### 3.2 Default resolution and the frequency-sized linspace

`_stretch_waveform` fills defaults only when an argument is `None`: `hop_length = n_fft // 4`, `win_length = n_fft`, `window = torch.hann_window(win_length)`. Explicit `hop_length` / `win_length` / `window` values are never checked against each other, so the window length is arbitrary. Then `phase_advance = torch.linspace(0, pi * hop_length, spec_f.shape[-2])[..., None]` - the length is the FREQUENCY bin count `n_fft//2+1`, NOT the frame count; the endpoint follows `hop_length`, and entry i is `pi*hop*i/(freq_bins-1)`.

### 3.3 Banker's rounding

`len_stretch = int(round(ori_len / rate))` uses Python's built-in `round` (ties-to-even), not `floor(x+0.5)`. With `n_steps=-12` giving `rate=2`: `ori_len=9` -> `4.5 -> 4`; `15` -> `7.5 -> 8`; `5` -> `2.5 -> 2`; `7` -> `3.5 -> 4`. That value is the `length` passed to `torch.istft`.

### 3.4 Trim / pad and view

`_fix_waveform_shape` compares `shift_len` with `ori_len`: too long -> `waveform_shift[..., :ori_len]`; too short -> `F.pad(..., [0, ori_len - shift_len])` with trailing zeros; finally `view(shape[:-1] + [ori_len])` restores the batch-leading dimensions. Because the last axis is normalised to `ori_len` first, the view target is always the original `waveform.size()`; both a 3-D leading shape `[2,3]` and a 1-D input with no leading dims must be rebuilt correctly.

### 3.5 gcd and ceil+cast in speed

`speed` first computes `src = int(factor*orig_freq)` and `tgt = int(orig_freq)` (TWO truncations), reduces them with `math.gcd`, and passes the reduced rates to `resample`. `factor=1.5, orig_freq=16000` -> `24000/16000` -> gcd 8000 -> `3/2`; `factor=0.7, orig_freq=44100` gives `30869.999...` through floating-point error, truncated to **30869** instead of 30870, so the gcd collapses to 1. Valid lengths use `torch.ceil(lengths * tgt / src).to(lengths.dtype)`: elementwise ceil first, then cast back to the original dtype; integer lengths produce an integer list, float lengths a float list of integral values, and `lengths=None` yields `None`.

## 4. Code skeleton

```python
def solve(mode, **kw):
    c = _cfg(kw)                      # apply all defaults
    L = _ledger(c)                    # one shared ledger for all modes
    if mode == "rate":
        return 2.0 ** (-n_steps / bins_per_octave)
    if mode == "pitch_rates":       # int() truncation, not round()
        return [int(sample_rate / rate), int(sample_rate)]
    if mode == "len_stretch":       # Python round is ties-to-even
        return int(round(ori_len / rate))
    if mode == "speed_rates":
        g = math.gcd(int(factor * orig_freq), int(orig_freq))
        return [src_raw, tgt_raw, g, src_raw // g, tgt_raw // g]
    ...                                # seven more modes
```

## 5. Key quirks and traps

1. `int(sample_rate / rate)` and `int(sample_rate * 2**(n/12))` are mathematically equal but can differ by one in floating point; the problem pins the former.
2. The linspace length of `phase_advance` comes from `spec_f.shape[-2]` (frequency bins); the injected spectrum's row count alone decides the output length.
3. `round` is ties-to-even, so `4.5 -> 4` but `7.5 -> 8`; `floor(x+0.5)` is wrong on half of the tie cases.
4. `speed` passes the GCD-REDUCED rates to `resample`, not the raw `int(factor*orig_freq)`.
5. `_fix_waveform_shape` trims/pads only the last axis; leading batch dims are rebuilt by `view`, and padding is always trailing zeros.
6. `n_steps` passes through `float()`, so fractional steps are legal; `bins_per_octave=0` raises `ZeroDivisionError`.
7. `stretched` (the istft output) is validated but never consumed at the orchestration layer: the resample step returns a separately injected `shifted` waveform.

## 6. Verification pipeline

Verbatim environment smoke test -> 12-segment probe `427/427 ALL GREEN` -> three-layer cross-check `13326 checks / 52 configs ALL GREEN` (the independent oracle uses `exp+log` instead of `pow`, a bit shift instead of floor division, `Fraction` steps, `Decimal ROUND_HALF_EVEN`, a hand-written Euclid loop, `Decimal ROUND_CEILING`, and index-driven chunk reshaping) -> judge driver `152/152 ALL ACCEPTED` -> mutation testing with 160 mutants, 13 documented true equivalents excluded, `147/147 CAUGHT / 0 SURVIVED / 0 ERROR`.

## 7. Scope cuts and registration

The judged domain contains no FFT, no complex-spectrum values, no window coefficients, no sinc convolution, no audio files and no model forward; waveforms, spectra and windows are all explicit inputs, with no torch / numpy / scipy / global RNG. The four bilingual docs are strictly line-aligned, and `register864.py` grows `audio` from 15 to 16, the problem count from 841 to 842 and `max_order` to 864, appending the 864 convention paragraph to both collection descriptions. Per the standing convention, no `git commit` is made until the user confirms.
