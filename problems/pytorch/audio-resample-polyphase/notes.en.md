# Solution Notes · Polyphase Resampling

## From "zero-stuff, convolve, decimate" to polyphase

The textbook recipe has three steps: upsample by `L` (insert `L-1` zeros), convolve with a low-pass kernel, keep every `M`-th sample. It is mathematically correct but wastes work twice:

- During convolution `L-1` out of `L` inputs are 0, so most multiply-accumulates hit zero.
- Only the `1/M` of outputs that survive decimation are ever needed.

Combining both observations, output `m` depends only on those `x[n]` whose index sits near `m·M`:

```
y[m] = Σ_n x[n] · h(m·M - n·L)
```

The index of `h` is a signed offset on the upsampled grid, and `n` only ranges over values that put `m·M - n·L` inside `[-c, c]`. Splitting the kernel by `m·M mod L` into `L` phases, each about `taps / L` long, is where the name comes from.

## Why the cutoff is 0.5 / max(L, M)

`h` lives on the upsampled grid, where Nyquist is 0.5 (cycles per sample). The two distortions each impose a bound:

- Upsampling creates `L-1` images outside the original band `[0, 0.5/L]`, so `fc <= 0.5 / L`.
- Downsampling requires the band to be squeezed to `0.5 / M` before decimation, so `fc <= 0.5 / M`.

Both hold exactly when `fc = 0.5 / max(L, M)`. Deriving it from some ratio like `down / up` always leaves out-of-band energy on one side: a metallic ring when upsampling, aliased hiss when downsampling.

## Phase alignment: the centre offset c

Array index `j ∈ [0, taps)` maps to signed offset `j - c` with `c = (taps-1)/2` (`taps` must be odd for `c` to be an integer). Read coefficients at index `m·M - n·L + c`.

Dropping the `+ c` is a subtle bug: outputs with `m·M ≡ 0 (mod L)` — the ones that land exactly on input samples — shift by `c` samples, even and odd phases move in opposite directions, and constants oscillate wildly while the impulse response merely looks "a bit odd". The constant-gain check exists to catch exactly this.

## Σh = L, not 1

Each phase is a subset of `h`. Normalising to `Σh = 1` makes every phase sum to `1/L`, shrinking the output by `L`. Normalising to `Σh = L` makes each phase sum to 1, so a constant input stays constant.

A truncated sinc is not an ideal low-pass, so that sum ripples by a few parts per thousand: about 0.7% at `taps = 63`, shrinking as `taps` grows.

## Output length and boundaries

`out_len = ((T-1)·L) // M + 1` emits only outputs whose main sample `m·M / L` lies in `[0, T-1]`, so the final upsampled output lines up with the final input sample. Outside the signal everything is zero, no padding, which leaves roughly `c / L` transient samples on each end.

Production implementations (e.g. `scipy.signal.resample_poly`) pad the edges to remove that transient; this problem deliberately specifies zero extension because its behaviour is unique and reproducible elementwise.

## Complexity

The naive version convolves at the upsampled length `T·L`, O(T · L · taps). The polyphase version touches only `taps / L` samples per output and emits `T·L/M` outputs, about O(T · taps / M) — **independent of L**, so a larger upsampling ratio costs nothing extra. Both wastes are removed together, for an overall speedup of roughly `L·M`.

## Related problems

- `audio-stft-frame`: the first frequency-domain transform after resampling.
- `audio-mel-filterbank`: the same "redistribute energy onto a new grid" idea.
- `audio-preemphasis-framing`: the entry point of the pipeline.
