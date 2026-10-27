# Audio Front-End: Polyphase Integer-Ratio Resampling

## Problem

Speech front-ends demand a fixed sample rate (Whisper 16 kHz, some acoustic models 8 kHz), while input may come from a 44.1 kHz microphone or a 48 kHz screen recording. **Integer-ratio resampling** is the workhorse case: target / source = `L / M` with small integers (16k → 48k is 3/1, 16k → 24k is 3/2).

The naive recipe is "insert zeros to upsample by `L`, low-pass filter, then keep every `M`-th sample", but after zero insertion `L-1` out of every `L` samples are zeros, so most multiply-accumulates multiply by zero. **Polyphase decomposition** splits the filter into `L` phases and touches only the samples that exist, cutting per-output work from `taps` to `taps / L`:

```
y[m] = Σ_n x[n] · h(m·M - n·L)
```

`h` is a windowed-sinc low-pass designed on the **upsampled** grid with centre offset `c = (taps-1)/2`, evaluated only where `|m·M - n·L| <= c`.

Three constants that break implementations:

- **Cutoff**: `fc = 0.5 / max(L, M)` (cycles per upsampled sample). Taking the larger of `L` and `M` is what suppresses both the upsampling images and the downsampling aliasing.
- **Prototype gain**: normalise so `Σh = L`, which makes every phase sum to 1 and reconstructs constants exactly.
- **Output length**: `out_len = ((T-1)·L) // M + 1`, emitting only outputs whose main sample lies inside the signal; outside the signal everything is zero (no edge padding).

## Task

Implement `solve(x, up, down, taps=63, beta=8.0)`:

- Inputs: `x` `(T,)`; `up = L`, `down = M` positive integers; `taps` an odd integer `>= 3`; `beta` the Kaiser parameter.
- Prototype: `fc = 0.5 / max(up, down)`, `h(j) = 2·fc·sinc(2·fc·j) · kaiser(taps, beta)[j+c]` for `j = -c … c`, rescaled so `Σh = up`.
- Output: `y[m] = Σ_n x[n]·h(m·down - n·up)`, summing only terms with `|m·down - n·up| <= c`; `out_len` as above.
- Return `(out_len,)` float32, computed in float64.

## Example

```
x = [1.0, 0.5, -0.5, 0.25], taps = 7 (c = 3), beta = 5

up = 2, down = 1 (2x upsampling, 7 outputs)
[1.014410, 0.754998, 0.507205, -0.009878, -0.507205, -0.129126, 0.253602]

up = 1, down = 2 (2x downsampling, 2 outputs)
[0.631391, -0.065841]

with the default taps = 63 the even outputs restore the input exactly:
[1.0, 1.082523, 0.5, -0.256106, -0.5, -0.145531, 0.25]
```

## Constraints

- **Phase alignment**: the kernel is centred at `j = 0`, so outputs with `m·down ≡ 0 (mod up)` land exactly on input samples and only the rest are interpolated. Forgetting to subtract `c` shifts even and odd outputs in opposite directions and makes constants oscillate violently.
- **The cutoff uses `max(up, down)`**, not some ratio like `down / up`; getting it wrong aliases when downsampling and images when upsampling.
- **`Σh = up`, not 1**: that is what makes each phase sum to 1. Normalising to 1 shrinks every output by a factor of `up`.
- **No edge padding**: outside the signal is zero, so the first and last `~c / up` outputs carry transients. That is the specified convention, not a bug.
- **`taps` must be odd**: only then does the centre land on an integer and the `c` arithmetic stay exact.
- **Invalid inputs raise `ValueError`**: not 1-D, `up/down < 1`, even `taps` or `taps < 3`, `beta < 0`, empty `x`.
