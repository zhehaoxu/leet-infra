# Implementation notes: 866 mel / linear filterbank scale ledger

## 1. Verbatim environment

The five functions are extracted from `functional.py` lines 425-633 via AST and exec-ed into one shared namespace with `torch.set_default_dtype(torch.float64)`. No proxy, no RNG, no audio IO. `_call` uses `warnings.catch_warnings(record=True)` to capture warnings and exceptions.

## 2. Reference solution

Pure Python using only `math`. A symmetric `linspace` reproduces torch bit-for-bit; `_safe` folds exceptions into `{type,msg}`; `_fbank` produces the matrix, warnings and the full ledger at once; all ten modes map one-to-one onto the official `comp`.

## 3. Independent oracle

`Fraction` rational `linspace`, `log1p` / `expm1` / `pow` transforms, a geometric piecewise triangle and set-membership column census. 69 configurations x 10 modes give 37578 comparisons with zero differences.

## 4. Constants and judge cases

121 official constants plus 123 `check_*` functions: all ten default modes, every break point of both scales, error dicts, triangular matrices, floor division, the four `norm` x `mel_scale` combinations, warning text, exact `linspace` and end-to-end recomputation.

## 5. Mutation and equivalence

168 mutants, 160 CAUGHT; 8 genuine equivalents are proved individually by `equiv866.py` over 270 configurations x 10 modes (21600 comparisons, zero differences). The hardening focus is exact equality assertions and the two silent branches of `linear_fbanks`.

## 6. Duplicate check and scope

Zero hits for the five functions; 342 binds Slaney to normalization, 852 is the Kaldi / VTLN compatibility layer, while 866 judges scale intermediates, `linear_fbanks` and the warning. The torch `max()` detail at `n_freqs=0` is outside the input domain.

## 7. Delivery and convention

Land 11 files, selftest in both modes, the bilingual check, register 844 problems, short guard and full guard with an isolated `--basetemp`, then backfill the log. No git commit until the user confirms.
