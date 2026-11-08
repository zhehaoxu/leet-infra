# torchaudio spectral post-processing ledger: the per-batch cutoff and packed_channels of amplitude_to_DB, the power convention of DB_to_amplitude, the cross-correlation denom of compute_deltas, sliding_window_cmn window recursion and its slice typo, and the frame/window dual normalization and DFT framing of spectrogram

## 1. Background

This problem reconstructs the **numerical conventions** of five spectral post-processing routines from `torchaudio.functional`, rather than calling the library. The judge environment has no `torchaudio` and no `torch` dependency is allowed. Every mode must be a deterministic pure-Python numeric ledger whose inputs are nested lists and whose outputs are nested lists of the same shape (or the compound dictionary specified below).

### Official source location

The upstream target is `pytorch/audio main::src/torchaudio/functional/functional.py` (2511 lines). Only seven top-level functions are used: `spectrogram`, `_get_spec_norms`, `_get_complex_dtype`, `amplitude_to_DB`, `DB_to_amplitude`, `compute_deltas`, and `sliding_window_cmn`. `_get_spec_norms` decides the two normalization booleans, and `_get_complex_dtype` maps a real dtype to its complex counterpart.

### Verbatim execution environment

`ast.get_source_segment` extracts the **verbatim source** of these seven functions (signature, docstring, and comments all preserved), which is then `exec`-ed into one shared namespace after injecting `torch`, `math`, `warnings`, `Tensor`, and typing names. Not a single official line is modified. The driver only converts nested lists into `float64` tensors, records the exact `torch.stft` call arguments, and splits complex results into `real` / `imag`. All comparisons run in `float64` so that `float32` quantization cannot interfere.

## 2. Conventions

### Inputs and outputs

The single entry point is `solve(mode, **kw)`. Required arguments and defaults for the five modes are listed below. Note that the padding mode of `deltas` is called `pad_mode` externally (to avoid colliding with the dispatch parameter `mode`); it corresponds to the third parameter `mode` of the official `compute_deltas`.

```text
solve("db",          x, multiplier, amin, db_multiplier, top_db=None)
solve("db_inverse",  x, ref, power)
solve("deltas",      specgram, win_length=5, pad_mode="replicate")
solve("cmn",         specgram, cmn_window=600, min_cmn_window=100,
                     center=False, norm_vars=False)
solve("spectrogram", waveform, window, n_fft, hop_length, win_length,
                     power=2.0, normalized=False, center=True,
                     pad_mode="reflect", onesided=True, pad=0,
                     return_complex=None)
```

Each mode returns a dictionary: `out` holds the numeric result, `shape` holds its shape, and the remaining fields are metadata tied directly to the mechanism (for example `cutoffs`, `windows`, `kernel`, `frame_norm`). The judge compares both values and metadata, so metadata must agree with the actual computation and must never be hardcoded.

### Numerical precision

Use Python `float` (IEEE double) throughout. Use `math.log10` for logarithms and `math.cos` / `math.sin` or an equivalent `cmath.exp` for the DFT; either `math.fsum` or plain summation is fine as long as the result agrees with the official `float64` output within a `1e-8` relative tolerance. No global RNG is allowed: every perturbation, window, and sequence is passed in explicitly.

## 3. mode 1: db

### Formula

`amplitude_to_DB` computes `x_db = multiplier * log10(clamp(x, min=amin))` and then subtracts `multiplier * db_multiplier` in place. `multiplier=10` is for power spectra and `20` for amplitude spectra; `amin` must be positive, and `db_multiplier` is usually the `log10` of a reference value.

```text
x_db = multiplier * log10(max(x, amin)) - multiplier * db_multiplier
packed_channels = shape[-3] if ndim > 2 else 1
num_batches     = prod(shape[:-3]) if ndim > 3 else 1
chunk           = packed_channels * freq * time
cutoff[b]       = max(vals[b * chunk : (b+1) * chunk]) - top_db
out             = max(x_db, cutoff[b])
```

### top_db and packed_channels

Clamping happens only when `top_db` is not `None`. The official code reshapes the tensor to `(-1, packed_channels, F, T)` with `packed_channels = shape[-3] if ndim > 2 else 1`, then takes `amax` over `(-3,-2,-1)` per batch. Hence inputs of 4D or more get **one independent cutoff per batch**, while 3D inputs (where `packed_channels` is the channel count) and 2D inputs get a **single** cutoff. The result must report `cutoffs` (`None` when `top_db` is `None`), `packed_channels`, and `num_batches`, and the length of `cutoffs` must equal `num_batches`.

### Domain constraints

`x` must have at least two dimensions, be rectangular, non-empty, and contain only numbers; `multiplier`, `amin`, `db_multiplier`, and `top_db` must be numbers, with `amin > 0`. Violations raise `ValueError` with a message naming the constraint that was broken.

## 4. mode 2: db_inverse

### Formula

`DB_to_amplitude` is `ref * (10^(0.1*x))^power`. `power=1` restores a power spectrum (paired with `multiplier=10`), and `power=0.5` restores an amplitude spectrum (paired with `multiplier=20`). Therefore `solve("db", ...)` followed by `solve("db_inverse", ...)` must round-trip the original spectrum under either pairing. The returned dictionary echoes `ref` and `power`.

### Domain constraints

`x` must have at least one dimension (a 1D list is allowed) and contain only numbers; `ref` and `power` must be numbers.

## 5. mode 3: deltas

### Cross-correlation and denom

`compute_deltas` requires `win_length >= 3` and sets `n = (win_length - 1) // 2`. **Even** window lengths are legal, and for those `n` differs from `win_length // 2`. The official denominator is `n*(n+1)*(2n+1)/3`, equal to `2 * sum(i^2 for i in 1..n)`. The kernel is `arange(-n, n+1)` and `conv1d(groups=channels)` is a **cross-correlation**: `d[t] = sum_j kernel[j] * padded[t+j] / denom`, matching the docstring's `sum_i i*(c[t+i] - c[t-i]) / (2*sum i^2)`. For a rising sequence, interior deltas must be positive.

```text
n      = (win_length - 1) // 2
denom  = n * (n + 1) * (2 * n + 1) / 3
kernel = [-n, ..., 0, ..., n]
padded = replicate_or_constant_pad(row, n)
d[t]   = sum_j kernel[j] * padded[t + j] / denom
```

### Padding modes

`pad_mode="replicate"` repeats the boundary value `n` times on each side; `"constant"` pads with zeros. They must differ at boundary frames but agree on interior frames (as long as the window does not touch the boundary). The input is first `reshape(1, -1, T)`, so all leading dimensions are flattened into channels and restored afterwards. The result reports `n`, `denom`, `kernel`, `mode`, and `num_channels`, where `mode` is the padding mode actually used.

### Domain constraints

`win_length` must be an `int` and `>= 3`; `pad_mode` must be `replicate` or `constant`; `specgram` must have at least two dimensions.

## 6. mode 4: cmn

### Window schedule

`sliding_window_cmn` interprets the last two dimensions as `(num_frames, num_feats)` and maintains a `[window_start, window_end)` interval per frame. Without centering the window lies to the left; with centering it is centered on the current frame. A negative start is clamped to 0 and shifts the end right; when not centered and `window_end > t`, `window_end = max(t+1, min_cmn_window)`; when the end runs past the frame count, the start is shifted back and clamped to 0.

```text
if center:  ws = t - cmn_window // 2 ; we = ws + cmn_window
else:       ws = t - cmn_window      ; we = t + 1
if ws < 0:  we -= ws ; ws = 0
if not center and we > t: we = max(t + 1, min_cmn_window)
if we > num_frames: ws -= we - num_frames ; we = num_frames ; ws = max(ws, 0)
window_frames = we - ws
out[t] = x[t] - cur_sum / window_frames
```

### Recursion and norm_vars

The first frame accumulates the frame sum and squared sum over `[ws, we)` in one pass; later frames remove the `last_window_start` frame when the left edge advances and add the `last_window_end` frame when the right edge advances. With `norm_vars=True` the row is multiplied by `variance^-0.5`, where `variance = cur_sumsq/T - (cur_sum/T)^2`; when `window_frames == 1` the row is set to zero directly. The result reports `windows` (a per-frame `[ws, we]` list), `num_frames`, `num_feats`, and `norm_vars`.

### Official quirks

The official first-frame slice is written as `specgram[:, window_start : window_end - window_start, :]`, using the window length instead of `window_end` as the upper bound. But this branch only runs when `last_window_start == -1` (that is, `t=0`), and the first frame's `window_start` is always clamped to 0, so `window_end - window_start == window_end`: **the typo is numerically unobservable**. A genuinely observable quirk: for a 2D input with a single frame, the trailing `squeeze(0)` reduces the output from `(1, F)` to `(F,)`, i.e. to a 1D list. The docstring also describes the input as `(..., time, freq)` and the output as `(..., freq, time)`, contradicting what the source actually does with the last two dimensions.

### Domain constraints

`cmn_window` and `min_cmn_window` must be positive integers; `center` and `norm_vars` must be `bool`; `specgram` must have at least two dimensions.

## 7. mode 5: spectrogram

### Framing and DFT

First apply `pad` samples of constant zero padding on both sides; if `center=True`, pad another `n_fft//2` on each side according to `pad_mode` (reflect uses `x[p], x[p-1], ...` on the left and `x[-2], x[-3], ...` on the right). When `win_length < n_fft` the window is zero-padded in the center: `left = (n_fft - win_length)//2`, `right = n_fft - win_length - left`. Each frame is multiplied by the window and transformed with an `n_fft`-point DFT; `onesided=True` keeps only `n_fft//2+1` bins. The output layout is `[freq][time]`, with complex results split into `{"real": ..., "imag": ...}`.

```text
x = pad(waveform, pad, "constant")
if center: x = reflect_or_constant_pad(x, n_fft // 2, pad_mode)
win = zero_pad_centered(window, n_fft, win_length)
seg[t] = [x[t * hop_length + j] * win[j] for j in range(n_fft)]
z[k][t] = sum_j seg[t][j] * exp(-2*pi*i*k*j/n_fft)
if frame_norm:  z /= sqrt(n_fft)
if window_norm: z /= sqrt(sum(window^2))
power is None -> (real, imag) ; power == 1.0 -> abs(z) ; else -> abs(z)^power
```

### Normalization branches

`_get_spec_norms` maps `True` and `"window"` to `window_norm` (divide by `sqrt(sum(window^2))`, using the **original** window rather than the padded one; both have the same squared sum), maps `"frame_length"` to `frame_length_norm` (passed to `torch.stft` as `normalized`, equivalent to dividing by `sqrt(n_fft)`), and leaves both false for `False`. An invalid string raises `ValueError`; any other type raises `TypeError`. A non-`None` `return_complex` only emits a deprecation warning and does not change the result, so the returned `warnings` list must have length 1.

### power branches

`power is None` returns complex values; `power == 1.0` returns `abs(z)`; any other number returns `abs(z)**power`. `waveform` supports 1D and 2D (`(channel, time)`); for 2D the output shape is `[channel, bins, frames]`. The result also reports `bins`, `frames`, `frame_norm`, `window_norm`, `normalized`, `window_padded`, `pad`, `center`, `pad_mode`, `onesided`, `power`, and `warnings`.

### Domain constraints

`waveform` must be 1D or 2D and contain only numbers; `window` must be a non-empty 1D list whose length equals `win_length`; `n_fft` and `hop_length` are positive integers and `0 < win_length <= n_fft`; `pad` is a non-negative integer; `center` and `onesided` are `bool`; `pad_mode` is `reflect` or `constant`; `power` is a number or `None`. Reflect requires `n_fft//2 < len(x)`, the frame count must be positive, and the window energy must be positive.

## 8. Official quirk checklist

1. The `top_db` cutoff is per batch, but 2D/3D inputs have only one batch.
2. `compute_deltas` uses cross-correlation rather than convolution; for even `win_length`, `n` comes from `(win_length-1)//2`.
3. The `sliding_window_cmn` slice-upper-bound typo is unobservable because the first frame's start is always 0.
4. A 2D single-frame CMN input is reduced in rank by `squeeze(0)`.
5. The CMN docstring contradicts the source about the meaning of the last two dimensions.
6. `spectrogram`'s `return_complex` argument is deprecated and only emits a warning.
7. `window_norm` uses the original window's squared sum; `frame_length_norm` goes through `stft(normalized=True)`.

## 9. Common mistakes

- Treating the 4D `top_db` cutoff as a single global value, or leaving `packed_channels` out of `chunk`.
- Using `win_length // 2` instead of `(win_length - 1) // 2`, which breaks even window lengths.
- Implementing deltas as a convolution (flipping the kernel), which reverses the sign.
- Forgetting the `min_cmn_window` branch, or treating `window_end` as a window length.
- Starting the reflect left segment at `x[p-1]` (it must be `x[p]`), which breaks the first frame's DFT.
- Treating `normalized=True` as `frame_length_norm`, or computing the squared sum from the padded window.
- Laying out complex output as `[time][freq]`, or confusing `power=1.0` with `power=2.0`.
- Hardcoding metadata fields so that they disagree with the numeric convention.

## 10. Self-check

The judge cases cover the formulas, domain constraints, metadata, and quirks of all five paths. The reference solution must pass twelve official probes, a three-layer comparison (official / reference / independent oracle), and full mutation testing. Any unknown mode must raise `ValueError` with a message containing `unknown mode`.
