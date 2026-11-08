# torchaudio functional waveform-operations ledger: mu-law companding and int64 truncation, convolve operand swapping with original-length slicing, the DFT product of fftconvolve, center cropping in _apply_convolve_mode, SNR scaling and the lengths quirk of add_noise, and phase accumulation with polar reconstruction in phase_vocoder

## 1. Background

This problem rebuilds, as a pure-Python numeric ledger, the nine top-level **waveform-side** functions of `torchaudio.functional`, rather than calling the library. The judge environment has no `torchaudio` and must not depend on `torch` or `numpy`; every mode is a deterministic pure-Python ledger over nested lists returning the composite dictionaries specified here. It is delimited from the order-855 spectral post-processing ledger: order 855 covers magnitude spectra, DB conversion, deltas, CMN and spectrogram, while this problem covers only `mu_law_encoding`, `mu_law_decoding`, `convolve`, `fftconvolve`, `_check_shape_compatible`, `_check_convolve_mode`, `_apply_convolve_mode`, `add_noise` and `phase_vocoder`.

### Official source

The source is `pytorch/audio main::src/torchaudio/functional/functional.py` (2511 lines, already fetched for order 855 and stored at `.workbuddy/scratch/captum855/functional.py`). The mu-law pair lives around lines 668-735, the convolution family at lines 2189-2315, `add_noise` from line 2317, and `phase_vocoder` at lines 732-802. The source is unchanged: `ast.get_source_segment` extracts each function body byte-for-byte, including its signature, docstring and comments.

### Verbatim execution harness

The nine functions are `exec`'d into one shared namespace `NS`; `NS` receives only `math`, `typing`, `warnings`, `torch` and a few typing aliases, and no official function body is mocked. The driver converts nested lists to `float64` tensors (`complex128` for complex spectra), so every numeric constant in this problem follows IEEE double precision. `mu_law_decoding` implicitly demotes a non-floating input via `to(torch.float)` to float32; the judged numeric path explicitly re-wraps the int64 codes as a `float64` tensor before decoding, so that implicit demotion is not the source of any expected value.

## 2. Conventions

### Inputs and outputs

The single entry point is `solve(mode, **kw)`. The six modes, with their required arguments and defaults, are listed below. `conv_mode` corresponds to the official `mode` parameter (the name avoids colliding with `solve`'s dispatch argument):

```text
solve("mu_law",        x, quantization_channels)
solve("convolve",      x, y, conv_mode="full")
solve("fftconvolve",   x, y, conv_mode="full")
solve("slice_mode",    full, x_length, y_length, conv_mode="full")
solve("add_noise",     waveform, noise, snr, lengths=None)
solve("phase_vocoder", real, imag, rate, phase_advance)
```

### Numerical precision and invalid inputs

Everything uses Python `int` / `float`; nested lists are flattened and rebuilt in C order. Broadcasting follows NumPy: trailing axes are aligned and every dimension must be equal or one of them must be 1. There is no global random source; waveforms, noise, SNR values and phase advances are all explicit inputs. An unknown mode raises `ValueError` containing `unknown mode`; every invalid input raises `ValueError` whose message names the violated contract (for example `quantization_channels must be an integer`, `The operands must be the same dimension`, `rate must be a positive number`). Validation order is itself part of the contract: when two inputs are invalid simultaneously, the one checked first wins.

## 3. Mode 1: mu_law

### Encoding and decoding formulas

Let `mu = quantization_channels - 1` and `L = log1p(mu)`. Encoding is `sign(x) * log1p(mu*|x|) / L`, linearly mapped into `[0, mu]`, then `0.5` is added before the `int64` cast. The official `.to(torch.int64)` truncates toward zero, and the mapped value is non-negative, so `+0.5` behaves as round-half-up. Decoding maps an integer code `e` back into `[-1,1]`:

```text
mu  = qc - 1
L   = log1p(mu) = log(1 + mu)
x_mu = copysign(log1p(mu * abs(x)) / L, x)
e    = int((x_mu + 1) / 2 * mu + 0.5)
dx   = (e / mu) * 2 - 1
dec  = copysign((exp(abs(dx) * L) - 1) / mu, dx)
```

Endpoint behaviour: `x = -1 / 0 / 1` encodes to `0 / qc//2 / qc-1`; decoding the midpoint yields a tiny positive number (about `8.62e-05` for `qc=256`) rather than exact zero; `x=0.3, qc=256` encodes to 228 and decodes to `0.30633397981616217`. The encoded result preserves the nested shape of the input and its elements are Python `int`.

### Domain constraints

`quantization_channels` must be an integer `>= 2`, and `bool` is not accepted as an integer (avoiding the `isinstance(True, int)` trap); `x` must be a non-empty numeric array with every sample in `[-1, 1]`. Violations raise `ValueError` containing `quantization_channels must be an integer`, `at least 2`, `x must be scaled to between -1 and 1`, or `x must be a non-empty numeric array`, respectively.

## 4. Mode 2: convolve

### Direct convolution and operand swapping

The official implementation uses `torch.nn.functional.conv1d` with `weight.flip(-1)` to obtain a **true convolution** (not cross-correlation). The `full` result always has length `N + M - 1`. Crucially, when the last dimension of `x` is shorter than that of `y`, the official code swaps the operands before convolving (true convolution is commutative, so values are unchanged), yet the final `_apply_convolve_mode(result, x_size, y_size, mode)` still receives the **original, pre-swap lengths**. Therefore `same` has the original `x_length` and `valid` has `max-min+1`; neither switches to `y`'s length because of the swap. The `swapped` field must faithfully record whether a swap occurred; equal-length operands are never swapped.

```text
full = [ sum_i a[i] * b[k-i] for k in 0..N+M-2 ]   (0 <= k-i < M)
valid: target = max(N, M) - min(N, M) + 1 ; start = (N+M-1 - target) // 2
same : target = N (original x_length)      ; start = (N+M-1 - N) // 2
out  = full[..., start : start + target]
```

### Leading-dimension broadcast

Both operands must have the same rank, and every leading dimension (all but the last) must be equal or one of them must be 1; otherwise `The operands must be the same dimension` or `Leading dimensions of x and y are not broadcastable` is raised. When valid, the leading dimensions are broadcast to a common shape `lead`, flattened into `B = prod(lead)` rows, convolved row by row, and rebuilt as `lead + [L]`. The result also reports `num_signals = max(1, prod(lead))` and `leading_shape`.

## 5. Mode 3: fftconvolve

`fftconvolve` computes `rfft(x) * rfft(y)` at `n = N + M - 1` points and then `irfft`, matching the direct method numerically (floating error within `1e-8`). Crucially it does **not** swap operands: even when `x` is shorter the original order is kept and `swapped` is always `False`. The slicing rules are identical to the direct method (again using the original `x_length` / `y_length`). This problem reproduces the path with pure-Python DFT / IDFT; missing high-frequency samples are equivalent to zero padding, so the `irfft` output must be real `float` values, not complex numbers whose imaginary part is merely close to zero.

## 6. Mode 4: slice_mode

This mode isolates `_apply_convolve_mode`: given an explicit `full` result and the two original lengths, it returns the center-cropped `out` together with `start_idx` and `target_length`. `full` returns the input unchanged; `valid` uses `max-min+1` rather than any other variant of `|N-M|`; `same` always uses `x_length`, even when `x` was the shorter operand. Centering uses floor division `// 2`, so an odd full length biases the start to the left. For multi-dimensional `full` only the last axis is sliced, and the start index must be passed down unchanged through recursion.

## 7. Mode 5: add_noise

### Energy and scale

The last dimensions of `waveform` and `noise` must match, and the rank of `snr` (and of optional `lengths`) must equal the rank of the waveform's leading dimensions; leading dimensions are unified into `lead` by broadcasting. For each row `b`:

```text
Es = sum_i waveform[b][i]^2   (i < length[b])
En = sum_i noise[b][i]^2      (i < length[b])
original = 10 * (log10(Es) - log10(En))
scale    = 10 ** ((original - snr[b]) / 20)
out[b]   = waveform[b] + scale * noise[b]
```

Equivalently `scale = sqrt(Es/En) * 10^(-snr/20)`; every 20 dB increase in SNR shrinks the scale by a factor of 10. A non-positive `Es` or `En` raises `energy_signal and energy_noise must be positive`.

### The official lengths quirk

`lengths` is used **only for computing the energies** (samples past the length count as zero), while the addition still uses `scale.unsqueeze(-1) * noise` with the **unmasked original noise** and the unmasked original `waveform`. Consequently the padded region contributes no energy but still receives scaled noise. For example, with `waveform=[1,2,3,100]`, `noise=[1,1,1,1]`, `snr=0`, `lengths=3`, we get `Es=14`, `En=3`, `scale=sqrt(14/3)`, and the fourth output is `100 + scale` rather than `100`. In addition, conflicting leading dimensions of `snr` / `lengths` must be rejected explicitly rather than silently broadcast into a larger shape.

## 8. Mode 6: phase_vocoder

### Time axis and phase accumulation

For `rate == 1.0` the official function **returns the input directly**, with no polar round-trip, so `real` / `imag` are the original input objects and the frame count is unchanged. Otherwise:

```text
time_steps = arange(0, T, rate)          -> n = ceil(T / rate), t_j = j * rate
alphas     = time_steps % 1.0
phase_0    = angle(spec[..., :1])
padded     = pad(spec, [0, 2])
s0, s1     = padded[..., int(t)], padded[..., int(t) + 1]
phase      = angle(s1) - angle(s0) - phase_advance
phase      = phase - 2*pi*round(phase / (2*pi))     # wrap into [-pi, pi]
phase      = phase + phase_advance
phase      = cat([phase_0, phase[..., :-1]])         # shift by one frame
phase_acc  = cumsum(phase)
mag        = alphas * |s1| + (1 - alphas) * |s0|
out        = polar(mag, phase_acc)                   # cos / sin reconstruction
```

`phase_advance` has shape `(freq, 1)`; this problem reduces it to one scalar per frequency bin. `real` / `imag` must have identical shapes with at least two dimensions `(..., freq, frames)`. The output frame count `num_frames_out = n` is returned along with `time_steps`, `alphas`, `num_frames_in`, `freq` and `rate`. A non-positive or non-numeric `rate`, or a mismatch between the bin count and `phase_advance`, raises a `ValueError` with a specific keyword.

### Domain constraints

Mismatched `real` / `imag` shapes raise `real and imag must have the same shape`; insufficient rank raises `complex_specgrams must have (..., freq, frames) shape`; an invalid `rate` raises `rate must be a positive number`; a wrong number of phase advances raises `phase_advance must have one value per freq bin`. Batched leading dimensions are flattened into `B` independent groups and rebuilt as `lead + [freq, n]`.

## 9. Returned fields

```text
mu_law        : encoded, decoded, mu, quantization_channels, shape
convolve      : out, shape, full, full_length, mode, swapped,
                x_length, y_length, leading_shape, num_signals
fftconvolve   : (same fields as convolve, with swapped always False)
slice_mode    : out, shape, mode, start_idx, target_length
add_noise     : out, shape, scale, energy_signal, energy_noise,
                leading_shape, length
phase_vocoder : real, imag, shape, time_steps, alphas,
                num_frames_in, num_frames_out, freq, rate
```

## 10. Self-test and judging

The judge supplies 20 top-level `check_*(solve)` cases covering values, shapes, returned fields, domain constraints and validation order for all six modes, plus asymmetric three-dimensional leading broadcast, distinct last-dimension values under a singleton leading dim, no swapping for equal-length operands, a three-tap kernel, a positive center-crop start, `y_length=1`, a conflicting SNR leading dimension, bool array elements, the real dtype of the FFT output, and the numeric real/imaginary parts of the phase vocoder. Every expected value comes from the verbatim official harness and is cross-checked against the reference solution and an independent oracle.
