# torchaudio sinc-resampling ledger: gcd reduction and width, Hann/Kaiser windows with the float32 beta demotion, phase-interleaved convolution at stride=orig_freq, and the official quirk where a deprecated alias lands in the wrong branch

## 1. Background

`torchaudio.functional.resample` resamples a waveform with band-limited sinc interpolation. This problem splits the chain into five independently judged pure-Python ledgers: the parameter reduction ledger, the modified Bessel I0 series, kernel construction, phase-interleaved convolution, and the end-to-end pipeline.
This problem is separated by algorithm family from `audio-resample-polyphase` (344, integer-ratio polyphase resampling): that one uses a polyphase filter bank with integer decimation/interpolation, while this one builds a sinc kernel and runs `F.conv1d(stride=orig_freq)` over a phase matrix. It shares `functional.py` with 855, 857 and 858 but uses a disjoint set of functions.

### Official source

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  _get_sinc_resample_kernel    (lines ~1303-1400)
  _apply_sinc_resample_kernel (lines ~1403-1427)
  resample                    (lines ~1430-1490)
```

### Verbatim execution environment

The staging file `.workbuddy/scratch/torchaudio859/official859.py` uses `ast.get_source_segment` to extract the three top-level functions byte-for-byte from the 2511-line official file already downloaded for problem 855, and `exec`s them into a shared namespace. Only `math`, `typing`, `warnings`, `torch`, `Tensor`, `Optional` and `_CPU` are injected; no function body is mocked. The driver layer uses float64 throughout.

## 2. Conventions

### Inputs and outputs

```text
solve(mode, **kw) -> dict
mode = meta | i0 | kernel | apply | resample
```

The reference solution exposes the single entry point `def solve(mode, **kw)`. The judge file holds a set of top-level `def check_*(solve)` functions into which the judge injects `solve`. Judge cases must not use `from solution import ...`.
All inputs are nested Python lists. The implementation is pure Python: no `torch`, `torchaudio` or `numpy`, and no global RNG. Any leading dimensions are flattened in C order into equal-length rows, computed row by row and folded back with `lead + [tail]`.

### Channel axis and numeric precision

The official kernel tensor has shape `(new_freq, 1, 2*width+orig_freq)`, with the middle axis always 1. The judged convention drops it: the kernel matrices of `kernel` and `apply` are two-dimensional `(n, K)`, while `meta.shape` still reports the official three-dimensional shape.
Judged numerics are float64 throughout, matching the official path where `resample` passes `waveform.dtype` into the kernel builder. With `dtype=None`, the official `idx` is float64 while `torch.arange(0,-n,-1,dtype=None)/n` yields float32 from int64 division, and `kernels.to(torch.float32)` downcasts the result; the three dtype paths are observably different (max diff about 1.4e-08) and are excluded from the judged domain.
In the Kaiser branch, `beta_tensor = torch.tensor(float(beta))` is always float32: the denominator `torch.i0(beta_tensor)` is float32 while the numerator is float64. The judged convention reproduces the denominator as the exact float64 series re-quantised to f32; the Cephes polynomial behind `torch.i0(float32)` differs by a few ULP for some beta values, so the judged Kaiser domain uses the default beta or values that agree bit-for-bit (such as 3.0 and 12.0).

### Invalid inputs

Every invalid input raises a `ValueError` with a specific keyword: `must be a non-empty numeric array`, `Frequencies must be of integer type`, `must be an integer`, `must be at least`, `must be positive`, `beta must be a number`, `Invalid resampling method`, `kernels must be a 2-D`, `kernel shape must be`, `should be positive`, or `unknown mode`. Validation order and exact messages are part of the contract.

## 3. Mode 1: meta

### Reduction and kernel size

```text
g = gcd(orig_freq, new_freq)
o = orig_freq // g ; n = new_freq // g
base_freq = min(o, n) * rolloff
width     = ceil(lowpass_filter_width * o / base_freq)
K         = 2 * width + o
shape     = [n, 1, K]        # official tensor
scale     = base_freq / o
```

Note that `width` uses the reduced **orig_freq** (`o`) while `base_freq` uses `min(o, n)`; for 8 to 4 downsampling, `min(2,1)=1` and `base_freq` equals the rolloff itself.
`meta` also exposes `method`, `deprecated`, `replacement`, `window_branch`, `warnings`, `beta`, `beta_default_used` and `scale`; `beta` is the f32-quantised value (the default 14.769656459379492 becomes 14.76965618133545).

### Deprecated-alias quirk

The official code emits a deprecation warning for `sinc_interpolation` / `kaiser_window` but **never rewrites the local** `resampling_method`; the window branch tests the literal `resampling_method == "sinc_interp_hann"`, so every other name falls into the else (Kaiser) branch. In other words `sinc_interpolation` produces a Kaiser kernel even though the warning says "The default behavior remains unchanged" and points at `sinc_interp_hann`. This problem reproduces that behaviour faithfully.

## 4. Mode 2: i0

### Modified Bessel series

```text
I0(x) = sum_{k=0}^{inf} (x^2 / 4)^k / (k!)^2
term_0 = 1 ; term_k = term_{k-1} * (x^2/4) / k^2
i0          -> float64 series
i0_float32  -> f32(float64 series)
i0_beta_float32 = f32(I0(f32(beta_default))) = 271903.5625
```

For x up to 20 the series converges to about 1e-16 relative accuracy and agrees value by value with `torch.i0(float64)`. `values` may be a scalar or a nested list; `i0(0)=1` and I0 is even.

## 5. Mode 3: kernel

### The t grid and clamp

```text
idx[k] = k / o                  for k in [-width, width + o)
t[j,k] = (-j / n + idx[k]) * base_freq
t      = clamp(t, -W, W)        W = lowpass_filter_width
t     *= pi                     # after the window is evaluated
```

### Window, sinc and scale

```text
hann   : window = cos(t * pi / W / 2) ** 2
kaiser : window = I0(beta * sqrt(1 - (t / W) ** 2)) / I0(beta)
sinc   = 1.0 if t*pi == 0 else sin(t*pi) / (t*pi)
kernel[j,k] = sinc[j,k] * window[j,k] * (base_freq / o)
```

`kernel` returns the two-dimensional `kernels` (`[n, K]`) plus the ledger views `t`, `window` and `sinc`, together with `width`, `kernel_length`, `scale`, `beta_float32`, `i0_beta_float32` and `meta`.
At the +/-W clamp boundary the Hann window is not exactly zero but the floating residue `cos(pi/2)**2` of about 3.7e-33; the Kaiser window is `1/I0(beta)` of about 3.7e-6, clearly nonzero. Both are visible at the first and last taps of the kernel.

## 6. Mode 4: apply

### Padding, strided convolution and phase interleave

```text
padded      = [0] * width + waveform + [0] * (width + o)
num_frames  = length // o + 1
phase[j][s] = sum_k padded[s*o + k] * kernels[j][k]
flat        = [phase[j][s] for s in frames for j in phases]
target_len  = ceil(n * length / o)
out         = flat[:target_len]
```

`F.conv1d` is cross-correlation rather than true convolution, so the kernel is not flipped. The `(B, n, num_frames)` output is interleaved by `transpose(1,2).reshape(B,-1)`, truncated to `target_length`, then viewed back with the input leading dimensions.
`apply` returns `out`, `shape`, `target_length`, `num_frames`, `conv_length`, `padded_length`, `width` and `stride`; the kernel matrix must have shape `[n, 2*width+o]`.

## 7. Mode 5: resample

`resample` first rejects `orig_freq <= 0 or new_freq <= 0` (a guard that exists only in the orchestrator, not in the kernel builder), then **returns the input unchanged** when `orig_freq == new_freq`; otherwise it builds the kernel with the same parameters, calls `apply`, and returns `out`, `shape`, `target_length`, `identity` and `meta`.

## 8. Verification results

```text
PROBE  859 DONE ok=124 fail=0
VERIFY 859 DONE checks=727 fails=0 ALL GREEN
DRIVER 859 35/35 ALL ACCEPTED
MUT    859 196 total / 8 equivalent / 188 active / 188 CAUGHT
       0 SURVIVED, 0 ERROR, ALL MUTANTS CAUGHT
```

The first mutation round left 25 SURVIVED and one anchor-index ERROR. Six groups of hardening cases (defaults, exact error messages, bool/string types, the sinc column, and the identity short-circuit meta) plus eight equivalence rulings closed all of them.

## 9. Official quirk list

(1) Deprecated aliases only warn and are never renamed, so `sinc_interpolation` actually takes the Kaiser branch. (2) `beta_tensor` is always float32, so the `i0` denominator is float32 while the numerator is float64. (3) Hann is not exactly zero at the clamp boundary. (4) Kaiser is `1/i0(beta)` rather than zero at that boundary. (5) With `dtype=None`, `arange` is int64, its division demotes to float32, `idx` stays float64 and the final kernel is downcast to float32. (6) `orig_freq == new_freq` returns the original tensor object. (7) The `<=0` guard lives only in `resample`. (8) `target_length` uses `ceil(n*L/o)` rather than the convolution output length. (9) The warning claims the default behavior remains unchanged, contradicting (1).

## 10. Deduplication and reuse

Before landing, a full scan confirmed `max_order = 858`, that 859 is free, that all 836 orders are unique and that the slug is new. A library-wide search of `meta.json` for `sinc_resample`, `sinc_interp_kaiser`, `griffinlim` and `spectral_centroid` returned no hits. The boundary with 344 is the algorithm family; 855 (spectral post-processing), 857 (waveform operators) and 858 (NCCF pitch) share `functional.py` but use disjoint function sets.
