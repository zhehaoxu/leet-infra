# torchaudio Griffin-Lim phase-recovery ledger: norms / dtype routing, reflect-padded framed DFT, Hermitian mirroring with window-envelope overlap-add, pad taking effect only when length is given, and in-place tprev momentum with the +1e-16 divisor

## 1. Background

`torchaudio.functional.griffinlim` repeatedly reconstructs phase from a magnitude spectrum, while `inverse_spectrogram` turns a complex spectrum back into a waveform via `torch.istft`. This problem also rewrites the `torch.stft` / `torch.istft` primitives they depend on as an explicit pure-Python DFT plus overlap-add, and splits the chain into six independently judged ledgers: normalization routing, complex dtype mapping, forward framed DFT, Hermitian inverse DFT with OLA, the `inverse_spectrogram` orchestration, and the full Griffin-Lim iteration.
This problem shares `functional.py` with 855, 857, 858 and 859 but uses a disjoint set of functions. `mask_along_axis` is already covered by `asr-specaugment-mask` (order 693) and is therefore avoided. Searches for `griffinlim` and `inverse_spectrogram` returned no hits library-wide.

### Official source

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  inverse_spectrogram  (lines ~148-205)
  _get_spec_norms      (lines ~208-223)
  _get_complex_dtype   (lines ~226-233)
  griffinlim           (lines ~236-353)
```

### Verbatim execution environment

The staging file `.workbuddy/scratch/torchaudio860/official860.py` uses `ast.get_source_segment` to extract those four top-level functions byte-for-byte from the 2511-line official file downloaded for problem 855, and `exec`s them into a shared namespace. Only `math`, `typing`, `warnings`, `torch`, `Tensor`, `Optional`, `Union` and `_CPU` are injected; no function body is mocked. The driver uses float64 / complex128 throughout. To make the initial phase of `rand_init=True` an explicit argument, only the driver layer temporarily replaces `torch.rand` with a fixed complex matrix (`angles_override`); the official function body is untouched.

## 2. Conventions

### Inputs and outputs

```text
solve(mode, **kw) -> dict
mode = norms | complex_dtype | stft | istft | inverse | griffinlim
```

The reference exposes the single entry point `def solve(mode, **kw)`. The judge file holds a set of top-level `def check_*(solve)` functions into which the judge injects `solve`; `from solution import ...` is forbidden. All inputs are nested Python lists; the implementation is pure Python with no `torch`, `torchaudio`, `numpy` or global RNG.
★ The judged domain takes only two-dimensional `(freq, frames)` spectrograms; the batch axis is excluded. A complex matrix is expressed as two real matrices `real` / `imag`; `angles_init` gives the initial phase explicitly as a matrix of `[re, im]` pairs, replacing the official `torch.rand(complex128)` random initial phase. `rand_init=False` corresponds to an all-ones complex initial phase, and `rand_init=True` corresponds to passing the exact matrix that `torch.rand` produced under a fixed seed.
Judged numerics are float64 throughout. The reference uses `cmath.exp` complex exponentials for the DFT, while the independent oracle expands the Hermitian inverse DFT into real cosine/sine pairs and gathers frames per output sample instead of scattering; both agree with the official implementation to about 1e-15.

### Invalid inputs

Every invalid input raises a `ValueError` with a specific keyword (`norms` type errors raise `TypeError`): `must be a non-empty list of numbers`, `must contain only numbers`, `must be a non-empty 2-D list`, `must have at least one column`, `must be rectangular`, `must be an int`, `must be >=`, `must be a number`, `must be a bool`, `win_length must be <= n_fft`, `hop_length must be <= win_length`, `window length must equal win_length`, `freq dimension must be n_fft // 2 + 1`, `real and imag shapes must agree`, `Invalid normalized parameter`, `Input type not supported`, `Unexpected dtype`, `power must be > 0`, `momentum must be in range`, `reflect pad requires`, `waveform shorter than n_fft`, `window overlap add min`, `rebuilt frame count does not match specgram`, or `unknown mode`. Validation order and exact messages are part of the contract.

## 3. Mode 1: norms / Mode 2: complex_dtype

`_get_spec_norms` maps `normalized` to the two booleans `(frame_length_norm, window_norm)`. Strings accept only `"frame_length"` and `"window"`; boolean `True` is equivalent to the string `"window"`; `False` clears both; any other type raises `TypeError`. `_get_complex_dtype` recognises only `double`/`float64` -> `complex128`, `float`/`float32` -> `complex64` and `half`/`float16` -> `complex32`; everything else raises `ValueError`.

```text
normalized=False        -> (False, False)
normalized=True         -> (False, True)
normalized='window'     -> (False, True)
normalized='frame_length' -> (True, False)
double / float64 -> torch.complex128
float  / float32 -> torch.complex64
half   / float16 -> torch.complex32
```

## 4. Mode 3: stft (reflect padding + framed DFT)

★ Reflect padding for `center=True` **excludes the boundary sample**: the left side is `x[1:p+1][::-1]` and the right side is `x[-p-1:-1][::-1]` with `p=n_fft//2`; hence `[1..8]` under `n_fft=8` becomes `[5,4,3,2,1,2,3,4,5,6,7,8,7,6,5,4]`, and the first frame's DC is 24 rather than 20. `len(x) > p` is required. When `win_length < n_fft`, the window is zero-padded with `(n_fft-win_length)//2` zeros on the left and the remainder on the right.

```text
p = n_fft // 2
padded = x[1:p+1][::-1] + x + x[-p-1:-1][::-1]
T = 1 + (len(padded) - n_fft) // hop_length
X[k,t] = sum_n padded[t*hop+n] * w[n] * exp(-2*pi*i*k*n/n_fft)
k = 0 .. n_fft//2 ;  normalized=True -> X / sqrt(n_fft)
left_zero = (n_fft - win_length) // 2
```

The output contains `shape=[n_fft//2+1, T]`, the two ledger matrices `real` / `imag`, `n_frames`, `padded_length`, and `n_fft` / `hop_length` / `win_length` / `center` / `normalized`.

## 5. Mode 4: istft (Hermitian mirror + OLA + envelope division)

A frame `col` of length `K=n_fft//2+1` is mirrored into an `n_fft`-point Hermitian spectrum via `full = col + [conj(col[n_fft-k]) for k in K..n_fft-1]`, then inverse-DFT'd with a `1/n_fft` factor and reduced to its real part; `torch.fft.irfft` discards the imaginary parts of DC and Nyquist, which the real-part reduction reproduces automatically. Each windowed frame is accumulated into a buffer at stride `hop_length` while `w[n]^2` is accumulated into the envelope, and the waveform is divided by the envelope sample by sample.

```text
total = (T-1)*hop_length + n_fft
buf[t*hop+n] += irfft(col_t)[n] * w[n]
env[t*hop+n] += w[n] * w[n]
start = n_fft//2 if center else 0
end   = total - (n_fft//2 if center else 0)
size  = length if length is not None else end - start
out[j] = buf[start+j] / env[start+j]
normalized=True -> buf *= sqrt(n_fft)
i >= total -> out = 0.0 , env = 1.0  (zero padding)
min(env) < 1e-11 -> ValueError('window overlap add min')
```

★ Three official quirks: (1) `center=True` merely starts at `n_fft//2` and trims `n_fft//2` at the end; when `length` is given it takes exactly `length` samples from `start`, padding zeros beyond the buffer with envelope 1. (2) `normalized=True` multiplies the whole buffer by `sqrt(n_fft)` (whereas `stft` divides). (3) A symmetric Hann window is zero at both ends, so a `length` that pushes the output past the window edge makes the envelope zero and `torch.istft` raises `window overlap add min`; a periodic Hann window or no `length` is safe.

## 6. Mode 5: inverse (inverse_spectrogram orchestration)

It first computes `(frame_length_norm, window_norm)`; if `window_norm` is true it multiplies the whole complex spectrum by `sqrt(sum(window^2))` using the **unpadded** `win_length` window; `frame_length_norm` is then passed to istft as `normalized`; when `length` is given it passes `length + 2*pad` to istft and slices `waveform[pad:-pad]`. ★ `pad` takes effect only when `length` is given; `pad_mode` appears in the official signature but is never used.

```text
frame_length_norm, window_norm = _get_spec_norms(normalized)
scale = sqrt(sum(w^2)) if window_norm else 1.0
spec  = spec * scale
istft_length = length + 2*pad if length is not None else None
wave = torch.istft(spec, ..., normalized=frame_length_norm,
                   length=istft_length)
if length is not None and pad > 0: wave = wave[pad:-pad]
```

`normalized=True` / `'window'` multiplies by `sqrt(sum(w^2))`, while `'frame_length'` goes through `istft(normalized=True)` and multiplies by `sqrt(n_fft)`; with a rectangular window and `n_fft=8` the two happen to be numerically equal but the mechanisms differ. The output contains `out`, `shape`, `target_length`, `frame_length_norm`, `window_norm`, `window_scale`, `istft_length`, `pad` and `envelope_min`.

## 7. Mode 6: griffinlim (phase iteration)

```text
if not 0 <= momentum < 1: raise ValueError
effective = momentum / (1 + momentum)
mag    = specgram ** (1 / power)
angles = angles_init or all-ones complex
tprev  = scalar 0.0
repeat n_iter times:
  inverse = istft(mag * angles, length=length)
  rebuilt = stft(inverse)                 # center=True, reflect
  angles  = rebuilt
  if effective: angles = angles - tprev.mul_(effective)
  angles  = angles / (abs(angles) + 1e-16)
  tprev   = rebuilt                     # UNnormalised
return istft(mag * angles, length=length)
```

★ Five high-value quirks: (1) `momentum` is rewritten in place to `m/(1+m)` before use. (2) `tprev.mul_(m)` is in-place but `angles.div(...)` is **out-of-place**, so `tprev = rebuilt` stores the unnormalised complex spectrum; the reference must keep a copy taken before normalisation. (3) The `+1e-16` divisor prevents `0/0` for a zero-magnitude spectrum (with an all-zero initial phase the official output is exactly an all-zero waveform). (4) Both the iterative and final `torch.istft` calls hardcode `center=True`, and `torch.stft` hardcodes `center=True, pad_mode="reflect"`; with `length=None` the inverse yields `(T-1)*hop` samples and the rebuilt frame count is exactly `T`, but a small hop makes reflect padding require `len(inverse) > n_fft//2` or it errors. (5) With an explicit `length`, the frame count must satisfy `1 + length//hop == T` or `specgram * angles` fails to broadcast; the reference reports `rebuilt frame count does not match specgram` explicitly.

The output contains `out`, `shape`, `target_length`, `momentum_effective`, `n_iter`, `power`, `n_frames` and `envelope_min`.

## 8. Official quirk list

(1) Reflect padding excludes the boundary sample. (2) Zero padding for `win_length < n_fft` is left-biased. (3) Hermitian mirroring automatically discards imaginary DC/Nyquist parts. (4) Window-envelope division with the `1e-11` NOLA threshold. (5) An overlong `length` pads zeros with envelope 1. (6) `normalized` acts in opposite directions in stft and istft. (7) `True` equals `'window'` while `'frame_length'` takes a separate branch. (8) `pad` matters only with `length`, and `pad_mode` is never used. (9) Momentum becomes `m/(1+m)` first. (10) `tprev` stores the unnormalised spectrum and `mul_` is in-place. (11) The `+1e-16` normalisation. (12) `n_iter=0` still uses the `rand_init` initial phase. (13) The frame-count constraint linking `length` and `hop`.

## 9. Verification and landing

```text
probe860.py   : PROBE 860 DONE ok=105 fail=0
verify860.py  : 2960 checks ALL GREEN / 0 SKIP
driver860.py  : 45/45 ALL ACCEPTED
mut860.py     : 226 total / 4 equivalent / 222 active
                222 CAUGHT / 0 SURVIVED / 0 ERROR
selftest      : run + submit both ACCEPTED
precheck860.py: all PASS
```

`land860.py` lands 11 files under `problems/pytorch/au-torchaudio-griffinlim-phase-ledger/` and verifies the three core artifacts against `variants/pytorch/` with `filecmp`.
