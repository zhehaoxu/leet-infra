# Implementation notes: 867 create_dct ledger

## 1. Verbatim environment

`create_dct` is extracted from `functional.py` lines 636-667 via AST and exec-ed into one shared namespace with `torch.set_default_dtype(torch.float64)`. No proxy, no RNG, no audio IO, no warning. `comp` implements `matrix` / `norm_error` / `census` and maps one-to-one onto the reference.

## 2. Reference solution

Pure Python using only `math`. `pi_over_nm` is hoisted out of the loops (reproducing the official top-level evaluation so `n_mels=0` raises `ZeroDivisionError`); the built matrix is `(n_mfcc, n_mels)`, transposed to `(n_mels, n_mfcc)`; `norm=None` scales the whole matrix by 2, `norm="ortho"` first scales `dct[0]` (the built row 0) by `1/sqrt(2)` then the whole matrix by `sqrt(2/n_mels)`. The three modes map one-to-one onto the official `comp`.

## 3. Independent oracle

`verify867.py`'s third layer computes `cos` independently via a Taylor series (recurrence `term *= -x^2/((2k+2)(2k+1))`) with no reliance on `math.cos`; it then validates the algebraic property `D @ D.T == I` for a square `norm="ortho"` matrix. 192 configurations x 3 modes give 87272 comparisons with zero differences.

## 4. Constants and judge cases

31 official constants plus 37 `check_*` functions: all three default modes, both `norm` matrices, error dicts, transpose shape, the four corners, orthonormality, exact cosine and the `n_mels=0` / `n_mfcc=0` boundaries.

## 5. Mutation

42 mutants, 42 CAUGHT, zero claimed equivalent; the hardening focus is the transpose direction, the ortho-first column and the top-level `n_mels=0` error. No equivalence-proof script (zero survivors).

## 6. Duplicate check and scope

`create_dct` has zero hits; `audio-mfcc` only applies an orthonormal DCT-II as one MFCC-pipeline step, and 855's notes merely mention the extraction trick, while 867 judges the matrix and `norm` conventions. The `ZeroDivisionError` at `n_mels=0` and the empty matrix at `n_mfcc=0` are implementation details / degenerate inputs, explicitly excluded or treated as zero columns.

## 7. Delivery and convention

Land 11 files, selftest in both modes, the bilingual check, register 845 problems, short guard and full guard with an isolated `--basetemp`, then backfill the log. No git commit until the user confirms.
