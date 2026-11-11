# torchaudio create_dct ledger: the DCT-II cosine matrix, the n_mels x n_mfcc transpose, norm=None scaling every entry by 2.0, and norm="ortho" scaling the first column by 1/sqrt(2) then the whole matrix by sqrt(2/n_mels), with an exact error for an invalid norm

## 1. Background

`torchaudio.functional.create_dct` builds the DCT-II transform matrix used by MFCC. It only does `cos`, a transpose and two `norm` scalings: NO RNG, NO audio IO, NO library beyond torch, and NO warning. This problem reproduces it in pure Python as three independently judged modes (`matrix` / `norm_error` / `census`) and pins every value to verbatim official snapshots.
Library-wide duplicate check: the name `create_dct` has ZERO hits across every `meta.json`, `description*.md` and `notes*.md` under `problems`. `audio-mfcc` merely applies an orthonormal DCT-II as one step of the MFCC pipeline (with lifter / CMVN, a single `norm` variant, on a logmel spectrogram) and does NOT build the `create_dct` matrix in isolation; order 855's notes mention `create_dct` only as the extraction trick and never cover the function. Order 867 judges `create_dct`'s matrix and `norm` conventions directly, with a clean boundary against earlier problems.

### Official source

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  create_dct(n_mfcc, n_mels, norm)   (lines 636-667)

  if norm is not None and norm != "ortho":
      raise ValueError('norm must be either "ortho" or None')
  n = torch.arange(float(n_mels))
  k = torch.arange(float(n_mfcc)).unsqueeze(1)
  dct = torch.cos(math.pi / float(n_mels) * (n + 0.5) * k)  # (n_mfcc, n_mels)
  if norm is None:
      dct *= 2.0
  else:
      dct[0] *= 1.0 / math.sqrt(2.0)
      dct *= math.sqrt(2.0 / float(n_mels))
  return dct.t()   # (n_mels, n_mfcc)
```

`official867.py` extracts the top-level `create_dct` with `ast.get_source_segment` and execs it into one shared namespace; NO official body is modified. The only injected objects are real `torch`, `math`, `typing` and type aliases, with the global default dtype pinned to `float64` so `torch.cos` can be cross-checked against a pure-Python reference. The matrix is recomputed with real torch in `official867.py` and is the sole authority for judge constants.

## 2. Conventions

### Three modes

`solve(mode, **kw)` dispatches three ways: `matrix` returns the matrix decided by `n_mfcc` / `n_mels` / `norm`, `norm_error` judges whether an invalid `norm` raises (or a valid one returns `"NO_ERROR"`), and `census` judges source-level facts. Apart from `unknown mode` raising `ValueError`, every exception (including an invalid `norm` and `n_mels=0`) is folded into a `{type, msg}` dict whose text matches the official wording verbatim.

### Matrix and scaling

The built matrix is DCT-II: `dct[i][j] = cos(pi/n_mels * (n+0.5) * k)`, shaped `(n_mfcc, n_mels)`, where `n` is the mel-axis index and `k` the mfcc-axis index. The function returns `dct.t()`, so the RETURNED matrix is shaped `(n_mels, n_mfcc)`, not `(n_mfcc, n_mels)`. `norm=None` multiplies the whole matrix by `2.0`; `norm="ortho"` first multiplies the BUILT matrix's row 0 (`dct[0]`) by `1/sqrt(2)`, then the whole matrix by `sqrt(2/n_mels)`, and that built row 0 becomes the RETURNED matrix's COLUMN 0 after the transpose.

## 3. Four official quirks

(1) `norm` accepts only `None` / `"ortho"`, case-sensitive: any other value (including `"ORTHO"` / `""` / `"none"`) raises `ValueError` with the exact text `norm must be either "ortho" or None`. (2) `norm=None` multiplies the ENTIRE matrix by 2.0, every `i, j` with no exception.
(3) under `norm="ortho"` the `1/sqrt(2)` scaling lands on the RETURNED matrix's COLUMN 0 after the transpose, not on row 0; the `sqrt(2/n_mels)` scaling is global. (4) `n_mels=0` raises `ZeroDivisionError` IMMEDIATELY when `math.pi / float(n_mels)` is evaluated (not lazily per element) and is a degenerate input explicitly outside the problem domain; `n_mfcc=0` instead returns an empty matrix of shape `(n_mels, 0)` (zero columns) with no exception.

## 4. Judge ledger

`fixtures867.py` exports 31 constants from the verbatim official environment, and `cases867_checks.py` supplies 37 `check_*(solve)` functions: numeric tolerance comparisons plus per-element EXACT equality assertions for the matrix (the reference uses the same DCT-II formula and transpose as the official, with the global default dtype pinned to `float64`, measured within 1e-12 of `torch.cos(float64)`). Hardening also covers the shape being `(n_mels, n_mfcc)`, the corner `cos(0)=1 -> 2.0`, the transpose formula, orthonormality (`norm="ortho"` with a square matrix gives `D @ D.T == I`), error dicts and the `n_mels=0` / `n_mfcc=0` boundaries.

## 5. Verification pipeline

```text
official867.py smoke          3 modes, EXIT=0
probe867.py                   1390/1390 ALL GREEN (12 segments)
probe_solution867.py          35593 checks / 0 bad (2 layers)
verify867.py                  87272 checks / 192 configs / 0 fails
fixtures867.py                31 constants
driver867.py                  37/37 ALL ACCEPTED
mut867.py                     42 mutants / 0 equivalents / 42 CAUGHT
```

The third cross-check layer shares NO algorithm with the reference: it computes `cos` independently via a Taylor series (recurrence `-x^2/((2k+2)(2k+1))`), not `math.cos`; it then validates the algebraic property that a square `norm="ortho"` matrix satisfies `D @ D.T == I`, a fact independent of the cosine formula that catches any transpose direction, scaling or formula mistake. All 42 mutants are caught by `check_*`; there is no mutant claimed equivalent, and none is ever deleted just to make the suite green.

## 6. Engineering lessons and scope cuts

The first trap is the TRANSPOSE DIRECTION: the built matrix is `(n_mfcc, n_mels)` and the returned one is its transpose `(n_mels, n_mfcc)`, so `returned[i][j] = cos(pi/n_mels * (i+0.5) * j)`; swapping the two indices is an easy mistake, so the transpose-formula check and the orthonormality property must pin it from both sides. The second trap is that the ORTHO-FIRST scaling lands on COLUMN 0, not row 0 -- `dct[0]` is the built matrix's row 0, which becomes the returned matrix's column 0 only after the transpose; asserting the row would miss the semantics.
The third trap is that `n_mels=0` raises `ZeroDivisionError` from the TOP-LEVEL immediate evaluation of `math.pi / float(n_mels)`, not from lazy per-element evaluation; the reference must hoist this step out of the loops, otherwise `n_mels=0` wrongly returns an empty matrix instead of raising. The fourth trap is that `n_mfcc=0` returns shape `(n_mels, 0)` (zero columns), not `(0, n_mels)`, so the boundary check must treat it as zero columns.

## 7. Registration and guards

`register867.py` registers with minimal intrusion, growing `audio` from 18 to 19 and appending the 867 convention paragraph to both descriptions; verification expects `845 registered = 845 disk / max_order=867 / no dangling / no unregistered`. `todos.md` is updated to 845 problems / audio 19 / next batch from 868 and flips 867 from planned to landed. After landing, selftest runs in both `submit` and `run` modes, followed by the library-wide bilingual check, the short guard and a full guard with an isolated `--basetemp`. Per the standing convention, NO git commit is made until the user confirms.
