# Audio Front-End: STFT Magnitude with Center Padding

## Problem

After framing and windowing, an FFT per frame gives the short-time Fourier transform. Real code has far more parameters than the textbook formula:

- **`n_fft`** sets the frequency resolution (bin spacing `sr / n_fft`) and the analysis length. Speech front-ends often use 400 (25 ms at 16 kHz); Whisper uses 400 with hop 160.
- **`center`** decides whether to pad `n_fft // 2` samples on both sides. With padding, frame `t` is centred on original sample `t·hop` and the frame count becomes `1 + T // hop`; without it the first frame starts at sample 0 and the count is `1 + (T - n_fft) // hop`. **librosa pads by default, and so does `torch.stft`.**
- **`pad_mode="reflect"`**: padded samples are neither zeros nor edge copies but a mirrored fold of the waveform (the boundary sample is not repeated). Zero padding creates a step from 0 to the real value whose leakage contaminates the whole frame; edge copying invents a fake DC component; reflection keeps the signal continuous at the endpoints and leaks the least.

The window here is the **periodic** Hann (denominator `N`, not `N-1`): the DFT assumes periodic extension, so only the periodic window closes at the frame boundary.

## Task

Implement `solve(wave, n_fft, hop_len, center)`:

- Inputs: `wave` `(T,)`; `n_fft` even and `>= 2`; `hop_len ∈ [1, n_fft]`; `center` boolean.
- `center=True`: reflect-pad `pad = n_fft // 2` samples on both sides (`x[pad], x[pad-1], …` on the left, `x[T-2], x[T-3], …` on the right), `F = 1 + T // hop_len` frames.
- `center=False`: no padding, `F = 1 + (T - n_fft) // hop_len` frames.
- Window: `w[n] = 0.5 - 0.5·cos(2πn / n_fft)`, `n = 0 … n_fft-1`.
- Output `magnitude = |rfft(frame · w, n_fft)|`, shape `(F, n_fft//2 + 1)`, float32.

## Example

```
wave = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
n_fft = 4, hop_len = 2, center = True

pad = 2, reflected x = [0.3, 0.2 | 0.1 ... 0.9 | 0.8, 0.7]
F = 1 + 9 // 2 = 5 frames, periodic Hann(4) = [0, 0.5, 1, 0.5]

magnitude (5 frames x 3 bins):
[[0.300000, 0.100000, 0.100000]
 [0.600000, 0.316228, 0.000000]
 [1.000000, 0.509902, 0.000000]
 [1.400000, 0.707107, 0.000000]
 [1.700000, 0.900000, 0.100000]]
```

## Constraints

- **The frame count depends on `center`**: with `center=True` it is `1 + T // hop`, independent of `n_fft`; only `center=False` subtracts one frame length.
- **Reflection does not repeat the boundary sample**: the left pad is `x[1..pad]` reversed, not `x[0..pad-1]`. Writing an edge copy shifts everything by one sample.
- **The window is periodic**, so its last point is nonzero; the symmetric form changes each frame's energy and differs dramatically for small `n_fft` (4, 8).
- **Return the one-sided spectrum only**: `n_fft//2 + 1` bins, with no `×2` energy compensation.
- **`n_fft` must be even** (the Nyquist bin of `rfft` is only defined then); `hop_len` cannot exceed `n_fft`; `center=True` requires `T > n_fft//2` (reflection would be undefined). Invalid inputs raise `ValueError`.
