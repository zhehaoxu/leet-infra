# Solution Notes · Pre-emphasis, Framing, Windowing

## Pre-emphasise the whole signal first

Pre-emphasis is a first-order IIR (`y[t] = x[t] - a·x[t-1]`) depending on two adjacent samples. If you cut first and then filter each frame, sample 0 of every frame has no predecessor and must be special-cased to `y = x` — so each frame boundary is broken, and overlapping regions get filtered twice when `hop_len < frame_len`. The correct order is one pass over the full waveform, then slice.

## The frame-count formula and the tail

`F = 1 + (T - frame_len) // hop_len`. It encodes three things: the first frame starts at 0, the last frame's **start** satisfies `start + frame_len <= T`, and a tail shorter than a frame is dropped. Using `ceil` instead produces one extra out-of-range frame (padded or crashing).

When `hop_len > frame_len` frames leave gaps and not a single symbol of the formula changes.

## Symmetric vs periodic windows

Two conventions share the name Hann:

- Symmetric (`sym=True`): denominator `N-1`, exactly zero at both ends. Used for filter design and for cases like this one where each frame is analysed on its own.
- Periodic (`periodic=True`): denominator `N`, equal to the symmetric window of length `N+1` with the last point removed. Used for the STFT, because the DFT assumes periodic extension.

`torch.hann_window(N)` is **periodic** by default; `torch.hann_window(N, periodic=False)` is symmetric. This problem uses symmetric; the STFT problem in the same batch uses periodic — that is exactly where implementations get contaminated.

## Hamming does not end at zero

`0.54 - 0.46·cos(0) = 0.08`, so Hamming keeps 8% amplitude at both ends while Hann reaches 0. That difference is why endpoint assertions are a classic trap.

## Vectorised form

Three lines suffice: `starts = arange(F) * hop_len`, `idx = starts[:, None] + arange(N)[None, :]`, `frames = y[idx] * w[None, :]`. Gather-indexing instead of `unfold` also makes `hop_len > frame_len` correct for free, since `unfold` only supports overlapping windows.

## Complexity

Time O(T + F·N), space O(F·N) for the output. Pre-emphasis is one O(T) pass and the window is precomputed once.

## Related problems

- `audio-stft-frame`: framing followed by rfft, and it makes the symmetric-vs-periodic difference explicit.
- `audio-mel-filterbank`, `audio-mfcc`: the next two stages of this pipeline.
- `audio-resample-polyphase`: usually runs before this pipeline to unify sample rates.
