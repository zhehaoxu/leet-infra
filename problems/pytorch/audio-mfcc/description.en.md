# Audio Front-End: MFCC via Orthonormal DCT-II

## Problem

Adjacent Mel bands are highly correlated (the triangles overlap), so feeding `n_mels` features straight into a classifier is both redundant and hard to model. **MFCC** applies one DCT that rotates those correlated features into an approximately decorrelated cepstral domain; keeping the first `n_mfcc` dimensions drops fine detail while preserving the vocal-tract envelope.

Three details decide whether the result is right:

1. **DCT normalisation**: only the orthonormal version (`norm="ortho"`) preserves energy. `c[k] = s_k · Σ_n x[n] cos(πk(n+0.5)/M)` with `s_0 = sqrt(1/M)` and `s_k = sqrt(2/M)`. An unnormalised DCT (`s_k = 2`) yields a matrix off by an overall `sqrt(M/2)`; CMVN hides it, but c0's energy meaning is completely different.
2. **lifter**: multiply each cepstral coefficient by `1 + (L/2) sin(πk/L)` to attenuate high-quefrency coefficients (a smoothing pass in the cepstral domain that suppresses noise). One trap: at `L = 2` the weight of `k = 3` is exactly `1 + sin(3π/2) = 0`, so that coefficient is erased.
3. **CMVN** (cepstral mean and variance normalisation): z-score each coefficient along time to remove the fixed channel/device offset — standard for ASR. The standard deviation divides by `T` (population), not `T-1`.

## Task

Implement `solve(logmel, n_mfcc, lifter=0, cmvn=False)`:

- Inputs: `logmel` `(T, M)` log-Mel energies; `n_mfcc ∈ [1, M]`; `lifter >= 0` (0 means no lifting); `cmvn` boolean.
- Orthonormal DCT-II along the Mel axis: `c[k] = s_k Σ_n x[n] cos(πk(n+0.5)/M)`, `s_0 = sqrt(1/M)`, `s_k = sqrt(2/M)`.
- When `lifter > 0`: `c[k] *= 1 + (L/2) sin(πk/L)`.
- When `cmvn=True`: per coefficient `(c - mean_t) / (std_t + 1e-8)` with the population standard deviation.
- Output `(T, n_mfcc)` float32.

## Example

```
logmel = [[1.0, 2.0, 3.0, 4.0],
          [2.0, 2.5, 3.0, 3.5],
          [0.5, 1.5, 2.5, 3.5]]
n_mfcc = 4

MFCC (T=3, M=4)
[[ 5.000000, -2.230442,  0.000000, -0.158513]
 [ 5.500000, -1.115221,  0.000000, -0.079256]
 [ 4.000000, -2.230442,  0.000000, -0.158513]]

with lifter = 4, w = [1, 2.414214, 3, 2.414214]
the first row becomes [5.0, -5.384764, 0.0, -0.382683]
```

## Constraints

- **Meaning of c0**: with an orthonormal DCT, `c[0] = sqrt(M) · mean(x)` — the whole frame energy collapses to one scalar; a constant input leaves only c0 nonzero.
- **Energy conservation**: with `n_mfcc = M`, `Σ c² = Σ x²` — the fastest self-check that normalisation is right.
- **Truncation happens after the DCT**: computing all `M` coefficients and slicing, or computing only the first `n_mfcc` rows, must agree elementwise.
- **The lifter zero**: `L = 2` kills `k = 3`; larger `L` approaches the identity.
- **CMVN runs after the lifter**: lift first, then normalise; the reverse also normalises the lifter weights and the two no longer commute.
- **All-zero input** must not produce NaN under CMVN: `std + 1e-8` keeps the output at 0.
- **Invalid inputs raise `ValueError`**: not 2-D, `n_mfcc` out of range, `lifter < 0`.
