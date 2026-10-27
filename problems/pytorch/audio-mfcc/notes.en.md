# Solution Notes · MFCC

## Why the DCT decorrelates

Mel filters are overlapping triangles, so adjacent bands are strongly correlated and the covariance matrix is nearly tridiagonal. The DCT-II basis vectors are the discrete approximation of that filter set's eigenfunctions, so one orthonormal transform approximately diagonalises the covariance. That is also why MFCC uses a DCT rather than PCA: no statistics needed, the basis is closed-form.

## The three constants of orthonormality

`c[k] = s_k Σ_n x[n] cos(πk(n+0.5)/M)`:

- `s_0 = sqrt(1/M)` makes the c0 of a constant vector `sqrt(M) · mean` instead of `M · mean`.
- `s_k = sqrt(2/M)` for `k ≥ 1` makes the basis an orthogonal matrix, so `Σ_k c_k² = Σ_n x_n²`.
- Dropping the `1/sqrt(2)` correction (using `sqrt(2/M)` everywhere) adds an extra factor of two's worth of c0 energy.

`scipy.fft.dct(x, type=2, norm="ortho")` matches exactly; `norm=None` (the default) is unnormalised and differs by an overall `sqrt(2M)`, which elementwise comparison exposes immediately.

## What c0 and c1 mean

c0 is frame energy (overall spectral magnitude), c1 approximates spectral tilt, and c2 upward describe the finer envelope shape. ASR often drops c0 (it mostly tracks recording volume) and keeps c1..c12 — which is why MFCC is usually 13-dimensional.

## The lifter formula and its zero

`w[k] = 1 + (L/2) sin(πk/L)`, with `w[0] = 1`, `w[L] = 1` and a peak of `1 + L/2` at `k = L/2`. For small `L` some coefficient can even go negative or to zero: `L = 2` gives `w[3] = 1 + sin(3π/2) = 0`, while `L = 1` gives `w[1] = 1 + 0.5 sin(π) = 1`. The implementation needs no special case, but a test should hit it.

## Two CMVN traps

- **Degrees of freedom**: `std` is the population version (divide by `T`). The sample version (`T-1`) deviates noticeably for small `T` (say 3) and disagrees with librosa / Kaldi.
- **Zero variance**: a constant column has std 0, so add `1e-8`. All-zero input is the easiest edge to hit: a plain division yields NaN or inf.

## Complexity

The DCT is a matmul `(T × M) @ (M × n_mfcc)`, O(T · M · n_mfcc); `M` is typically 40–80 so the cost is negligible. Production code accelerates the DCT with an FFT (O(T · M log M)), worthwhile only when `M` reaches the thousands.

## Related problems

- `audio-mel-filterbank`: its log-mel output is this problem's input.
- `audio-stft-frame`: the upstream power spectrum.
- `audio-preemphasis-framing`: the entry point of the pipeline.
