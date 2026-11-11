# torchaudio Gaussian Frechet distance ledger: five shape guards and the chained-comparison quirk, the squared-mu term and the two traces, the real-part sum of principal square roots of eigvals(SX@SY), and the F = a + b - 2c assembly

## 1. Background

`torchaudio.functional.frechet_distance` (functional.py lines 2476-2511) is the kernel of FID (Frechet Inception Distance): given the means mx/my and covariances SX/SY of two multivariate Gaussians, it returns `||mx-my||^2 + Tr(SX+SY-2 sqrt(SX SY))`. The official body is only 35 lines with four non-default parameters, yet it writes the trickiest step as `torch.linalg.eigvals(sigma_x @ sigma_y).sqrt().real.sum()` - take the PRINCIPAL square root of every (possibly complex) eigenvalue FIRST, then sum the real parts. It is neither a symmetrised matrix square root nor a clamp-then-sqrt.
This problem reproduces the ledger in pure Python and splits it into seven independently judged modes. Library-wide searches for `frechet`, `FID`, `eigvals` and `sqrtm` in meta, description and notes return no hits; orders 855 and 857-862 already covered the spectral post-processing, waveform operators, NCCF, sinc resampling, Griffin-Lim, loudness gating and MVDR beamforming of the same file, with a disjoint set of functions. **Source correction**: `todos.md` previously described this function as a pure-DP discrete Frechet distance; it actually cites the Gaussian distance of Dowson and Landau (1982) and has nothing to do with curve Frechet or Levenshtein edit distance.

### Official source

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  frechet_distance            (line 2476, 35 lines, no defaults)
    guard 1..5                 shape / dimension validation
    a = (mu_x - mu_y).square().sum()
    b = sigma_x.trace() + sigma_y.trace()
    c = torch.linalg.eigvals(sigma_x @ sigma_y).sqrt().real.sum()
    return a + b - 2 * c
```

### Verbatim execution environment

`official863.py` extracts the `frechet_distance` definition from the pinned 2511-line upstream source with `ast.get_source_segment` and execs it into a shared namespace; the official body is not modified. All tensors are float64, so `eigvals` returns complex128. `ledger()` recomputes a / b / A / eigvals / sqrt / c / F line by line for probes and fixtures, and `sorted_eigvals()` normalises the implementation-defined LAPACK root order by `(re, im)`.

## 2. Conventions

### Inputs and outputs

- Four required arguments: `mu_x`, `sigma_x`, `mu_y`, `sigma_y`; each mu is a length-N 1-D list and each sigma an N-by-N 2-D list of int / float / bool leaves (bool casts to 1 / 0).
- The judged domain is N<=4 with means and covariances passed explicitly; no model forward, no audio IO, no global RNG, and no scipy / librosa / soundfile / numpy / torch dependency in the reference solution.
- Complex eigenvalues are `[re, im]` pairs; `eigvals` returns them sorted lexicographically by `(re, im)`, and real roots carry an exactly zero imaginary part.
- `charpoly` returns the monic descending-power coefficients `[1, c1, ..., cN]` of `det(lambda*I - A)`, where `A = SX @ SY`.
- Seven modes: `frechet` (the full F), `mu_term` (a), `trace_term` (b), `sqrt_trace_term` (c), `terms` (`[a,b,c]`), `charpoly` and `eigvals`.

### Invalid inputs

- Five shape guards raise in the official order: `Input mu_x must be one-dimensional`, `Input sigma_x must be two-dimensional`, `Each of sigma_x's dimensions must match mu_x's size.`, `Inputs mu_x and mu_y must have the same shape`, `Inputs sigma_x and sigma_y must have the same shape`.
- Ragged data, non-numeric leaves, non-list arguments, missing arguments and unknown modes each raise explicitly; the unknown-mode message must list all seven MODES.

## 3. Mode 1: frechet (full distance)

Returns `a + b - 2*c`, the only mode compared directly against the official return value. On the main fixture (N=3, mx=[1,-2,0.5], my=[0,1,-0.5]) it gives a=11.0, b=7.5, c=3.647581187475623 and F=11.204837625048754. Swapping `(mx,SX)` and `(my,SY)` leaves the result unchanged: `A=SX SY` and `SY SX` share a spectrum, and a and b are symmetric.

## 4. Modes 2-4: the three components

`mu_term` is `sum((x-y)**2)` and is completely independent of the covariances; `trace_term` is `tr(SX)+tr(SY)` and reads only the diagonal, so changing off-diagonal entries leaves b fixed but changes c and F through A; `sqrt_trace_term` is `sum(cmath.sqrt(lambda_i).real)` and is the most subtle term in this problem.

## 5. Mode 5: terms (triple)

Returns `[a, b, c]` in that fixed order. The three components and `frechet` must satisfy the assembly identity; this mode lets the judge pin each term separately instead of relying on one final scalar.

## 6. Mode 6: charpoly

The reference solution derives the elementary symmetric polynomials `e_k` from the power traces `tr(A^k)` via Newton identities, then assembles the monic descending coefficients as `coeff[k] = (-1)^k e_k`; N=1/2/3 uses the linear formula, the quadratic formula and Cardano, while N>=4 falls back to Durand-Kerner iteration. The independent oracle takes a different route with the Faddeev-LeVerrier matrix recursion `M_k = A M_{k-1} + c_{k-1} I`, sharing no code with the reference.

## 7. Mode 7: eigvals (complex spectrum)

Returns every eigenvalue of `A=SX@SY`, sorted by `(re, im)`. The LAPACK return order is implementation-defined, so the judged domain compares sorted spectra; `c` only depends on an order-invariant sum, so sorting does not affect the value. Roots of a real polynomial come in conjugate pairs, and `AB` and `BA` share a spectrum.

## 8. Principal square roots of complex eigenvalues

The official `.sqrt().real.sum()` order must not be swapped. With SX=[[1,2],[-2,1]] and SY=I the spectrum of `A` is `1 +/- 2j`; the principal-root real part is `sqrt((|z|+Re z)/2) = sqrt((sqrt(5)+1)/2) ~ 1.272`, so c ~ 2.544 and F ~ -1.088. Clamping the real parts before the square root would give the wrong c=2.0. Negative real roots -4 and -1 have purely imaginary principal roots `2j` and `1j` whose real parts vanish, hence c=0 and F=-3. The official code never checks positive definiteness, so F may be negative.

## 9. The chained-comparison quirk and shape guards

The third guard `sigma_x.size(0) != sigma_x.size(1) != mu_x.size(0)` is a Python chained comparison, i.e. `(s0 != s1) AND (s1 != n)`, NOT a three-way distinctness test. For a square sigma the first clause is always False and the whole guard short-circuits: a 3-by-3 sigma with a length-2 mu is accepted silently, because mu only appears in `mx-my` and never meets sigma in a matmul. Only a non-square sigma with `s1 != n` triggers this message; a non-square sigma with `s1 == n` also short-circuits and errors later at the downstream square-matrix check. The reference solution reproduces this with a literal chained comparison.

## 10. Official quirk list

- `F = ||delta mu||^2` for `SX=SY=S` holds only when the spectrum of S is non-negative (then `sqrt(S^2)=S`); with S=diag(-4,-1), `sqrt(S^2)=|S|` and `F(X,X) = -20`, not zero.
- The official code checks neither symmetry nor positive (semi)definiteness and does not symmetrise a `sqrtm`; a non-symmetric `A` that is similar to an SPD matrix still has a positive real spectrum.
- Shape messages embed `torch.Size([...])` text; `mu_y` and `sigma_y` have no independent ndim guard, so their rank errors surface as shape mismatches.
- The official function has no defaults, no dtype argument and no batch axis; the `:cite:dowson1982frechet` reference in its docstring is the Gaussian distance, not a discrete curve distance.

## 11. Example and landing

```text
solve('frechet',         mu_x=[3.0], sigma_x=[[4.0]],
                        mu_y=[1.0], sigma_y=[[1.0]])   -> 5.0
  a = (3-1)^2 = 4 ; b = 4 + 1 = 5 ; c = sqrt(4) = 2
  F = 4 + 5 - 2*2 = 5
```

Pipeline results: the verbatim environment smoke test passes, the 12-segment probe is 159/159 ALL GREEN, the three-layer cross-check is 2238 checks / 52 configs ALL GREEN, the judge driver is 107/107 ALL ACCEPTED, and mutation testing is 127/127 CAUGHT after excluding 10 documented true equivalents from 137 mutants. Eleven files land (four bilingual docs + meta + solution / starter / test_cases + the variants/pytorch triple), followed by dual-mode selftest, library-wide bilingual check, short guard and full guard.
