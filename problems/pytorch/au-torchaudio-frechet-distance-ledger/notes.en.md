# Implementation notes: torchaudio Gaussian Frechet distance ledger

## 1. Selection and duplicate check

863 = `au-torchaudio-frechet-distance-ledger`, with order 863 free, the slug unused, and 14 problems in the audio collection. `frechet`, `Frechet`, `FID`, `eigvals`, `sqrtm` and `Dowson` have no hits in library meta, description or notes (`frechet` appears only as a candidate in the 860 notes). Sourcing corrected an old `todos.md` claim: this is the FID Gaussian distance, not a discrete Frechet / DP computation and not Levenshtein edit distance, which is already taken by `ocr-rec-norm-edit-dis`.

## 2. Verbatim execution environment

`official863.py` extracts `frechet_distance` from the pinned 2511-line `.workbuddy/scratch/captum855/functional.py` with `ast.get_source_segment` and execs it into a shared namespace; the official body is neither modified nor mocked. `_rc` turns nested lists directly into float64 tensors (non-numeric leaves pass through so the official body raises its own AttributeError); `ledger()` recomputes intermediates with the exact official expressions, and `sorted_eigvals()` removes the implementation-defined LAPACK root order.

## 3. Pipeline results

The 12-segment probe is `159/159 ALL GREEN` (verbatim source, five guards, the chained-comparison truth table, a/b terms, A and spectrum, complex principal roots, the c term, assembly, degeneracies, census). The three-layer cross-check is `2238 checks / 0 fails / 52 configs ALL GREEN`: layer 1 is official LAPACK + torch.sqrt; layer 2 is the reference with Newton identities + closed-form Cardano + cmath.sqrt; layer 3 is an oracle using Faddeev-LeVerrier matrix recursion + Durand-Kerner iteration + a hand-rolled polar square root, plus an independent Denman-Beavers matrix square root for `Tr(sqrt(A))` whenever A has no negative real spectrum. No algorithm is shared. The judge reports `107/107 ALL ACCEPTED`.

## 4. Mutation testing and true equivalents

Round one: 137 mutants / 116 CAUGHT / 18 SURVIVED / 3 anchor conflicts. After six hardening cases (error messages must contain `torch.Size([2])` / `[1]` and `[3, 3]` / `[2, 2]`, tuple inputs, empty-mu shape mismatch, a non-numeric leaf inside sigma, and exactly-zero imaginary parts for real roots) and three occurrence-index fixes, the result is `127/127 CAUGHT / 0 SURVIVED / 0 ERROR`. Ten true equivalents are recorded in `mut863.py::EQUIVALENT` with proofs: `l-cp-order` (A commutes with its powers), `s-sqrt-neg` (conjugate-root permutation leaves Re sqrt unchanged), `v-return-swap` (a/b symmetry plus the AB/BA spectrum), `h-shape-descend` (`_pack` already guarantees rectangular data), `h-pack-cast` (`float()` is a no-op on numeric leaves), `h-pack-width` (ragged detection only needs some reference row), `r-n3-w-imag` (swapping w and w^2 merely exchanges two roots of a sorted spectrum), `r-n3-branch` (principal cube roots of a real cubic satisfy uv=-p/3), `k-den-skip` (the judged N=4 spectra are separated so no denominator approaches zero), and `k-ret-old` (the 1e-14 break threshold is far below the 1e-8 judge tolerance).

## 5. Scope cuts and engineering lessons

The judged domain is fixed at N<=4 with all means and covariances explicit; no model forward, no audio IO, and no scipy / numpy / torch dependency. The biggest trap is coefficient orientation: Newton identities naturally produce ascending-power coefficients while the root solver needs monic descending powers, which made the first cross-check entirely red; switching to `[(-1)^k e_k]` brought all 11 configurations within 1e-14. The second trap is triple roots: Durand-Kerner splits I3 by about 4e-6 in the imaginary direction, so the reference keeps closed forms for N<=3 and the independent oracle computes c through a Denman-Beavers matrix square root. The third trap is that `F(X,X)=0` holds only for a non-negative spectrum, so negative-definite and complex-spectrum inputs need dedicated cases.

## 6. Official quirks and duplicate-check conclusion

Same as sections 9-10 of `description.md`: the chained-comparison short circuit silently accepts a square sigma with a mismatched mu length; the official `.sqrt().real.sum()` takes the principal square root before the real part; no PSD check exists; and the shared AB / BA spectrum makes F symmetric. Additionally `torch.trace` uses the short diagonal of a non-square matrix, so non-square inputs fail downstream at `eigvals` rather than at trace; `F_BOOL` and `F_INT` show that bool / int leaves are silently cast by `torch.tensor(..., dtype=float64)`. Library-wide duplicate checks confirm the function set is fully disjoint from orders 855 and 857-862.

## 7. Registration and guards

`register863.py` registers with minimal intrusion, growing `audio` from 14 to 15 and appending the 863 convention paragraph to both descriptions; verification expects `841 registered = 841 disk / max_order=863 / no dangling / no unregistered`. `todos.md` is updated to 841 problems / audio 15 / next batch from 864. The library-wide bilingual check expects `0 error / 15 warning` (all 15 historical); the short guard uses an isolated `--basetemp`, and the full guard result is backfilled after its background run. Per the standing convention, no git commit is made until the user confirms.
