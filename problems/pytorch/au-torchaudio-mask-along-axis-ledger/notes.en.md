# 865 Problem notes: torchaudio SpecAugment mask ledger

## 1. Selection and duplicate check

Order 865, collection `audio` 16 -> 17, slug `au-torchaudio-mask-along-axis-ledger`. Sources are `_get_mask_param` (806-810), `mask_along_axis_iid` (813-882) and `mask_along_axis` (885-958) in `functional.py`. IMPORTANT SOURCE CORRECTION: `mask_along_axis` had been recorded as already covered by order 693, but 693 is a FunASR/espnet explicit segment list ledger entirely unrelated to these three torchaudio functions; `todos.md` has been corrected. Library-wide meta / description / notes checks confirm the function set is disjoint from orders 855 and 857-864.

## 2. Official source and verbatim environment

`official865.py` extracts the three top-level definitions with `ast.get_source_segment` and execs them into a shared namespace; no official body is modified. `torch` is a forwarding proxy whose `rand` alone is replaced by a recorder returning explicit caller-supplied tensors, while `Tensor` stays the real `torch.Tensor` so `isinstance(mask_value, Tensor)` remains live. `_interval`, `_mask_rows` and `_apply_ledger` re-derive the mask from the recorded rand calls and `_check_ledger` asserts it reproduces the verbatim official output, so the interval ledger is an independent re-check of the official result rather than a second implementation.

## 3. Mechanism ledger

The `p==1.0` shortcut and `min(mask_param, int(axis_length*p))` truncation; dim guards (shared >=2, iid >=3) with verbatim messages; the `[dim-2, dim-1]` axis set; a closed `p in [0,1]`; `mask_param<1` returning the original object; the shared `reshape([-1]+shape[-2:])` pack/unpack, dtype-less `torch.rand(1)` hence float32, the separate `min_value.long()+value.long()` casts, `unsqueeze(-1)` on the frequency axis and the dead width guard; and the iid leading-dims rand shape, two transposes, the `where` + `mask_value.repeat` Tensor branch and the absence of a width guard. On the pure-Python side `struct` quantizes float32 per scalar step and `math.trunc` reproduces `.long()`.

## 4. Three-layer verification and mutation

Twelve probe segments give `380/380 ALL GREEN`. The three-layer cross-check reports `22054 checks / 74 configs / 0 fails`: official runs the real verbatim environment, the reference is pure-Python recursion with `struct` quantization and `math.trunc`, and the independent oracle uses flat-index arithmetic, set membership, `Decimal ROUND_DOWN` and `ctypes.c_float`; no algorithm is shared. The judge driver reports `140/140 ALL ACCEPTED`.

Round one: 169 mutants / 154 CAUGHT / 14 SURVIVED / 1 ANCHOR-ERROR. The anchor error came from the shared guard message being split across two source lines and was fixed to a single-line anchor. Thirteen hardening cases left three survivors, each proved equivalent and recorded in `mut865.py::EQUIVALENT`: `h-pack-float` (the float cast in `_pack` is overridden by a second cast in both `_nest` base cases, and leaf validation happens BEFORE the cast), `i-shared-first` (the shared rand_shape is always `[1]` and the draw-size guard forces exactly one vdraw / sdraw element, so `[0] is [-1]`), and `e-cases-dimshared` (the `dim_shared` case uses a 1-D spectrum and the dim guard reports only the dimension count, so 3 or 4 elements give the same result). `equiv865.py` shows zero divergences over 1901 configurations x 10 modes = 19010 comparisons per mutant, including exception type and message. Final result: **169 mutants, 3 documented equivalents excluded, 166/166 CAUGHT, 0 SURVIVED, 0 ERROR**.

## 5. Scope cuts and engineering lessons

The judged domain excludes real audio IO, mel / STFT numerics and model forwards; spectra, rand draws and `mask_value` are explicit inputs, pure Python with no numpy / torch / scipy and no global RNG. The biggest trap is that the DEFAULT leading dims are too small: the default `(2,3,4)` spectrum has only 2 iid draws, so neither `0.31+0.17k` nor `0.13+0.29k` crosses 1.0 and the `% 1.0` modulo mutant survives; even n=4 from `SPEC4=(2,2,3,4)` is insufficient, so a `(2,3,3,4)` spectrum with n=6 leading dims is required. The second trap is that the `_map2` slice step satisfies `prod(shape[1:]) == prod(shape[:-1])` on `(2,2)`, so only asymmetric leading dims `(2,3)` expose it. The third trap is that the default `mask_value=0.0` lets a mutant hard-coding 0.0 in the frequency branch survive, so a frequency-axis case with `mask_value=-1.0` is mandatory; an integer fill value additionally needs a `float` TYPE assertion on the output leaves, because numeric comparison does not distinguish int from float.

## 6. Official quirks and duplicate-check conclusion

Same as section 5 of `description.md`: `int()` truncates toward zero while the docstring says floor; the dtype-less shared `torch.rand(1)` makes the interval float32 while iid follows the spectrum dtype; `end` sums two separately truncated values; the shared width guard is unreachable under legal draws and entirely absent in iid; `mask_param<1` returns the original object and consumes no draw; the Tensor `mask_value.repeat` happens after the transpose. Additionally, every `variant` other than the literal `"iid"` falls through to shared, negative axes are rejected by the axis set, and both endpoints of the closed `p` interval are legal. Library-wide checks confirm order 693 is unrelated and the function set is fully disjoint from orders 855 and 857-864.

## 7. Registration and guards

`register865.py` registers with minimal intrusion, growing `audio` from 16 to 17 and appending the 865 convention paragraph to both descriptions; verification expects `843 registered = 843 disk / max_order=865 / no dangling / no unregistered`. `todos.md` is updated to 843 problems / audio 17 / next batch from 866 and flips 865 from planned to landed. The library-wide bilingual check expects `0 error / 15 warning` (all 15 historical); the short guard uses an isolated `--basetemp` and the full guard result is backfilled after its background run. Per the standing convention, no git commit is made until the user confirms.
