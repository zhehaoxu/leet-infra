# Solution Notes · Mel Filterbank

## Order: equally spaced in mel, then back to Hz

There is only one correct flow: map `f_min` and `f_max` into the mel domain, cut `n_mels + 2` equally spaced points there, then map back to Hz. Doing it the other way (equal spacing in Hz, then mapping to mel) gives the low frequencies one or two filters and the high frequencies all the rest, because equal Hz spacing is denser and denser in the mel domain as frequency grows.

Why `n_mels + 2` and not `n_mels`: each triangle needs a left zero, a centre and a right zero, and adjacent triangles share edges, so `n_mels + 2` points in total.

## The two scales

HTK: `mel = 2595 log10(1 + f/700)`, inverse `f = 700(10^{mel/2595} - 1)`. Logarithmic everywhere, so low frequencies never degenerate.

Slaney: for `f < 1000`, `mel = f / (200/3)` (3 mel per 200 Hz); above, `mel = 15 + ln(f/1000) / (ln(6.4)/27)`. The constant `log(6.4)/27` makes frequency double every 27 mel past 1 kHz (6.4 being Slaney's chosen crossover ratio). The pieces meet at 1 kHz with a continuous first derivative, so there is no kink.

Both land near mel ≈ 15 at 1 kHz (HTK gives 1000.6, Slaney 15), but the absolute mel units differ, so you cannot switch only half the pipeline.

## Area normalisation

Where `2 / (hz[m+2] - hz[m])` comes from: in the continuous domain the triangle's integral over the power spectrum is roughly `width × height / 2`, so dividing by `width/2` yields an average power density. Without it white noise of equal strength gives larger mel values in wide high-frequency bands purely because they are wider. It also explains why Slaney mel values are three orders of magnitude smaller than HTK ones — irrelevant for training (log and standardisation follow), but it matters for readability and cross-implementation comparison.

## Endpoints and DC

The first triangle's left edge is `f_min`. With `f_min = 0` the bin at 0 Hz sits exactly on that edge with weight `(f - lo)/(ce - lo) = 0`; the last triangle's right edge is `f_max`, so the Nyquist bin is zero as well when `f_max = sr/2`. So both DC and Nyquist fall outside every filter — counter-intuitive, but exactly what librosa does, and an assertion point here.

## The context of `1e-10`

Whisper uses `log10(max(mel, 1e-10))` followed by `(x + 4) / 4`, which is this problem's `log` plus an affine rescale. The floor turns silence into a large negative number instead of `-inf`.

## Complexity

The filter matrix is O(n_mels · F) and the projection O(T · F · n_mels). Production code treats it as a sparse matrix or sums over each triangle's narrow bin range, bringing it down to O(T · F).

## Related problems

- `audio-mfcc`: adds an orthonormal DCT on top of log-mel.
- `audio-stft-frame`: where the power spectrum comes from.
- `audio-preemphasis-framing`: the first stage of the pipeline.
