# Solution Notes · STFT Magnitude

## What each parameter controls

| Parameter | Role | Typical trap |
| --- | --- | --- |
| `n_fft` | frequency resolution and window length | an odd value leaves the Nyquist bin undefined |
| `hop_len` | step between frames | larger than `n_fft` skips whole segments, forbidden here |
| `center` | whether to pad `n_fft//2` on both sides | once padded, `n_fft` disappears from the frame count |

## Two shapes of the frame-count formula

Padded: `F = 1 + T // hop`. Read it as: after padding the signal has length `T + n_fft`, the last frame starts at `T + n_fft - n_fft = T`, and the start must be a multiple of `hop` no greater than `T`, hence `F - 1 = T // hop`.

Unpadded: `F = 1 + (T - n_fft) // hop`, where the last start cannot exceed `T - n_fft`.

The difference is exactly the two padded frames; the middle frames are identical — which makes a direct assertion: dropping the first frame of the `center=True` result, the next `F_false` frames equal `center=False` elementwise.

## Exact semantics of reflection

The left side of `F.pad(x, (pad, pad), mode="reflect")` is `x[1], x[2], …` reversed (no `x[0]`); the right side is `x[T-2], x[T-3], …`. That is precisely "do not repeat the boundary sample". With `replicate` you would copy the first sample `pad` times, create a constant plateau at the frame edge and inflate the DC bin out of nowhere.

Require `pad < T` (i.e. `T > n_fft//2`): beyond that `reflect` repeats the fold, which is subtle and implemented inconsistently across libraries, so we forbid it.

## The periodic window is the STFT window

`w[n] = 0.5 - 0.5·cos(2πn/N)`, `n = 0..N-1`, so `w[N-1] = 0.5 - 0.5cos(2π - 2π/N) ≠ 0`. It satisfies `w[n] = w[n+N]`, matching the DFT's periodic extension, so "window × signal" does not blow up at frame boundaries. The symmetric form (denominator `N-1`) visibly weakens the last point for small `N`; for `N = 4` the periodic window is `[0, 0.5, 1, 0.5]` and the symmetric one is `[0, 0.75, 0.75, 0]` — an obvious difference.

## Scale of the magnitude

`|rfft|` depends on the window: a constant `A` after a periodic Hann gives a DC bin of `A·N/2` and `A·N/4` on each neighbour. That is a handy self-check: if your neighbours are zero you used a rectangular window.

## Complexity

Time O(F·n_fft·log n_fft), space O(F·n_fft) for the frame matrix plus O(F·n_fft/2) for the output. Production code uses real-input FFTs (e.g. `kiss_fftr`) and reuses one buffer for the frame matrix to avoid re-allocation.

## Related problems

- `audio-preemphasis-framing`: the previous stage, and it contrasts symmetric with periodic windows.
- `audio-mel-filterbank`: projects this power spectrum onto Mel bands.
- `audio-resample-polyphase`: the preprocessing step for multi-rate inputs.
