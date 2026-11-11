# Implementation notes: torchaudio MVDR beamforming ledger

## 1. Topic selection and duplicate check

Order 862 is free, the slug `au-torchaudio-mvdr-beamform-ledger` is unused, the library has 839 problems, max_order=861 and every order is unique. `mvdr`, `souden`, `rtf_power`, `beamforming`, `diagonal_loading`, `Tikhonov` and `_tik_reg` have zero hits in all meta.json and description / notes files; the only `psd` meta hits are `rl-sweet-rl-turn-critic` and `rl-opsd-turn-credit` (the OPSD acronym, unrelated to power spectral density). The reason for choosing it: within the 2511 lines of functional.py, orders 855 and 857-861 already cover spectral post-processing, waveform operators, NCCF, sinc resampling, Griffin-Lim and loudness gating, so the MVDR family (nine functions at lines 1799-2198) is the last large uncovered mechanism cluster, with deterministic ledgers for complex outer products, real-only Tikhonov loading, solve plus complex-trace normalisation, the power-iteration schedule and the conjugate einsum.

## 2. Verbatim execution environment

`official862.py` uses `ast.get_source_segment` to extract the nine top-level definitions (including `rtf_evd`) from `captum855/functional.py` and execs them into one shared namespace with `math`, `warnings`, `typing`, `torch`, `Tensor`, `Optional` and `Union` bound; no official body is changed. `C()`, `cx()`, `R()` and `py()` convert between `[re, im]` lists and complex128 / float64 tensors, and each `comp_*` driver self-checks an independently transcribed ledger with torch.allclose alongside the official result. `rtf_evd` is probe-only: torch 2.14.1 in `.venv-eval` supports complex `linalg.solve` / `linalg.eigh`, but eigenvector signs are not portable.

## 3. Pipeline results

Twelve probe segments give `PROBE 862 DONE ok=241 fail=0 ALL GREEN`. The three-layer cross-check gives `VERIFY 862 DONE checks=134334 fails=0 configs=18 ALL GREEN`; the third-layer oracle deliberately uses a different algorithm, representing complex numbers as `(re, im)` float pairs with hand-written multiply / divide and solving the linear systems with closed-form 2x2 / 3x3 adjugate inverses instead of LU elimination, comparing each field with both the verbatim official output and the pure-Python reference. `fixtures862.py` freezes official values into the constants of `fixtures862.txt`, and the judge cases reference only these hand-verified constants. The judge driver reports `73/73 ALL ACCEPTED`.

## 4. Mutation testing and true equivalents

Four rounds: the first gave 94 CAUGHT / 34 SURVIVED / 2 ERROR (the two errors came from adding an occurrence index to unique anchors, which corrupted the file); after extra hardening fixtures and 24 judge cases the second gave 123 / 7, the third 126 / 4; finally `check_unknown_mode` was strengthened to require the full MODES list (including `rtf_power`) in the error message and `check_reference_channel_bad_tensor_leaf` was added to pin `pack`'s own `reference_channel must be rectangular`, giving 128 / 2. Final result: `130 raw mutants, 2 documented equivalents excluded, 128 active: 128 CAUGHT, 0 SURVIVED, 0 ERROR / ALL MUTANTS CAUGHT`.
Both equivalents are proved. `h-unpack-bool`: `pack` rejects bool leaves through `_is_num`, and every `unpack` argument comes from pack output (complex / float) or locally constructed numeric ledgers, so the bool arm is unreachable. `l-elim-start`: `_solve` already sets `M[r][col] = 0j` before the elimination loop, so starting that loop at `col` instead of `col+1` merely writes a nonzero value into the strict lower triangle; later pivot searches read only columns `col' > col`, and back substitution reads only `j >= i` plus the RHS columns, so strict-lower entries are never read and X is bit-identical.

## 5. Stripping decisions and engineering lessons

The leading `...` batch axis is stripped everywhere, fixing the judged domain at `(F, C, C)` / `(F, C)` / `(C, F, T)`; complex values are `[re, im]` pairs and bool is not a number. `torch.linalg.solve` is reproduced by complex Gaussian elimination with partial pivoting, and singular matrices raise explicitly instead of silently dividing by zero. Excluding `rtf_evd` in favour of an explicit RTF input is the biggest trap in this family: LAPACK eigenvector sign / phase is implementation-defined and would break cross-implementation comparisons. Also note the docstring versus signature default mismatches in `psd` and `_tik_reg` (always follow the signatures), and that the `.real` denominator of `rtf` and the complex trace of `souden` are the two conventions most easily conflated.

## 6. Official quirks and duplicate-check conclusion

Same as section 10 of `description.md`: two incorrect docstring defaults, the `withou` / `martrix` typos, complex dtype checks degenerating into leaf validation, and the `v[..., -1]` selection with unjudgeable eigh. Further details: the official `psd` einsum conjugates its right operand and sums over `dim=-3`; `_tik_reg` multiplies epsilon by reg before adding eps; `rtf_power` uses `n_iter - 2` loop steps plus a final `psd_s` multiply; `apply_beamforming` conjugates `w`. Library-wide duplicate checks confirm the function set is fully disjoint from orders 855 and 857-861.

## 7. Registration and guards

`register862.py` registers with minimal intrusion, growing `audio` from 13 to 14 and appending the 862 convention paragraph to both descriptions; verification expects `840 registered = 840 disk / max_order=862 / no dangling / no unregistered`. `todos.md` is updated to 840 problems / audio 14 / next batch from 863. The library-wide bilingual check expects `0 error / 15 warning` (all 15 historical); the short guard uses an isolated `--basetemp`, and the full guard result is backfilled after its background run. Per the standing convention, no git commit is made until the user confirms.
