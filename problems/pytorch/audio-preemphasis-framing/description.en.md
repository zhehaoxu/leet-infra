# Audio Front-End: Pre-emphasis, Framing, and Windowing

## Problem

The first layer of every speech model (Whisper, HuBERT, wav2vec2) is not a neural network but three fixed signal-processing stages: **pre-emphasis**, **framing**, and **windowing**. They cut a 1-D waveform into short-time stationary chunks so that the following FFT means something.

- **Pre-emphasis**: speech has far less energy at high frequencies (the glottal pulse spectrum decays roughly at −6 dB/oct). A first-order difference `y[t] = x[t] - a·x[t-1]` lifts the highs so the spectrum is flatter and has a smaller dynamic range. `a` is usually 0.97.
- **Framing**: speech is statistically stationary over 20–40 ms, so we cut frames of `frame_len` samples with a hop of `hop_len`. When `hop_len < frame_len` adjacent frames overlap, so information at frame boundaries is not wiped out by the window.
- **Windowing**: cutting a block is multiplying by a rectangular window whose sidelobes only reach −13 dB, leaking energy from strong tones far away. Hann (−31 dB) or Hamming (−43 dB) suppresses that leakage.

The order is not negotiable: pre-emphasis is per-sample and must happen before cutting, otherwise every frame's first sample would need the previous frame's last sample.

## Task

Implement `solve(wave, preemph, frame_len, hop_len, window="hann")`:

- Inputs: `wave` `(T,)` mono waveform; `preemph` the coefficient `a ∈ [0, 1)`; `frame_len` / `hop_len` in samples; `window` one of `"hann"` / `"hamming"` / `"rect"`.
- Pre-emphasis: `y[0] = wave[0]`, `y[t] = wave[t] - a·wave[t-1]` for `t ≥ 1`.
- Framing: **no padding at all**, `F = 1 + (T - frame_len) // hop_len` frames, frame `f` is `y[f·hop_len : f·hop_len + frame_len]`.
- Windowing: multiply each frame by `w`, Hann is `0.5 - 0.5·cos(2πn/(N-1))`, Hamming is `0.54 - 0.46·cos(2πn/(N-1))` with `N = frame_len`; `w = [1]` when `N = 1`.
- Outputs: `y` `(T,)` and `frames` `(F, frame_len)`, both float32 (compute in float64).

## Example

```
wave    = [0.5, -0.25, 0.75, 0.0, -1.0, 0.25, 1.0, 0.5, -0.5, 0.125]
preemph = 0.97, frame_len = 4, hop_len = 3, window = "hamming"

pre-emphasis: y[0] = 0.5
              y[1] = -0.25 - 0.97*0.5   = -0.735
              y[5] =  0.25 - 0.97*(-1.) =  1.22
              y = [0.5, -0.735, 0.9925, -0.7275, -1.0, 1.22, 0.7575, -0.47, -0.985, 0.61]

frame count F = 1 + (10 - 4) // 3 = 3, window w = [0.08, 0.77, 0.77, 0.08]

frame0 = y[0:4]  * w = [ 0.04  , -0.56595,  0.764225, -0.0582 ]
frame1 = y[3:7]  * w = [-0.0582, -0.77   ,  0.9394  ,  0.0606 ]
frame2 = y[6:10] * w = [ 0.0606, -0.3619 , -0.75845 ,  0.0488 ]
```

## Constraints

- **Symmetric window convention** (denominator `N-1`), matching `torch.hann_window(N, periodic=False)`; the periodic form (denominator `N`) is the STFT convention and differs at the last point — do not mix them up.
- **The frame count floors**: a tail shorter than one frame is dropped, not zero-padded. `T < frame_len` must raise instead of returning empty output.
- **`hop_len` may exceed `frame_len`**: frames then leave gaps; the formula is unchanged.
- **Pre-emphasis runs exactly once**: do not re-apply it to an already emphasised segment.
- **Invalid inputs raise `ValueError`**: not 1-D, `preemph ∉ [0, 1)`, `frame_len < 1`, `hop_len < 1`, `T < frame_len`, unknown window name.
