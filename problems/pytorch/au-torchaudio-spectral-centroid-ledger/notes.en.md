# Implementation notes: 868 spectral_centroid ledger

## 1. Verbatim environment

`spectral_centroid` is extracted from `functional.py` lines 1257-1299 via AST and exec-ed into one shared namespace with `torch.set_default_dtype(torch.float64)`. No proxy, no RNG, no audio IO, no warning; the only substitute is a `spectrogram` spy that returns an explicit magnitude spectrogram and records 12 kwargs. `comp` implements eight modes mapping one-to-one onto the reference.

## 2. Reference solution

Pure Python using only `math`. A symmetric `linspace` (back half computed from `end`) matches torch bit-for-bit; `_shape_of` reproduces ragged detection; `_weighted` reproduces the `(F,1)` broadcast; `_sum_freq_dim` reproduces `sum(dim=-2)`; `_div` normalizes `0/0` to the string `"nan"` and gives `+/-inf` for a nonzero numerator over zero; the output shape is `shape[:-2] + shape[-1:]`. All eight modes map one-to-one onto the official `comp`.

## 3. Independent oracle

`verify868.py`'s third layer independently recomputes the grid / weighted sums / numerator / denominator with exact `Fraction` rationals and validates 99 algebraic properties (single-bin impulse centroid, interval bounds, column-scaling invariance); 84 configurations give 43488 comparisons with zero differences and no shared floating-point formula with the reference.

## 4. Constants and judge cases

33 official constants plus 50 `check_*` functions: 13 frequency grids, 8 centroids, 3 ledgers, 2 spy kwargs, zero frame / shapes / errors / census, plus three hardening checks (fresh census dict, +/-inf branch, shapes exception branch).

## 5. Mutation and equivalence

155 mutants minus 4 proven equivalents leaves 151 all CAUGHT: the 4 equivalent survivors are proven by `equiv868.py` with 188 configs x 9 modes = 6768 full-output comparisons; the other 4 survivors were judge blind spots caught after hardening. Five anchor errors were fixed with explicit occurrence indices.

## 6. Duplicate check and scope

`spectral_centroid` has only 2 hits, both exclusion notes in orders 859 / 860, with no real coverage; order 855 judges `spectrogram` itself and the post-processing chain, while 868 only injects spectrogram as an explicit input spy and judges kwargs plus centroid semantics. `n_fft<=1` triggers torch's single-bin broadcast / empty-linspace implementation detail and is explicitly outside the input domain (same treatment as order 866's `n_freqs=0`).

## 7. Delivery and convention

Land 11 files, selftest in both modes, the bilingual check, register 846 problems, short guard and full guard with an isolated `--basetemp`, then backfill the log. No git commit until the user confirms.
