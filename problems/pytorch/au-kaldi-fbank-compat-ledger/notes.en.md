# 852 Solution Notes: Kaldi-Compatible Frontend Ledger

## 1. Official source retrieval

### Two source files and create_dct

Two files were pulled from the `main` branch of `pytorch/audio`: `src/torchaudio/compliance/kaldi.py` (813 lines) and `src/torchaudio/functional/functional.py`. The `mfcc` path of `kaldi.py` depends on `torchaudio.functional.create_dct`, which is absent from the judging environment, so `create_dct` was extracted verbatim from the official `functional.py` via AST into a standalone `create_dct.py`, avoiding any bias from a hand-written DCT.

### Literal execution environment

`official_kaldi.py` was built: it injects a fake `torchaudio` and `torchaudio.functional`, places the official `create_dct` into `_DCT_NS`, attaches it to the fake module, and then `exec`s `kaldi.py` verbatim. A smoke test confirmed 36 symbols in the namespace, all five window types, `create_dct(4,4)` with a first column of `0.5`, and mel / inverse-mel being mutual inverses. The probes therefore run the real official implementation.

## 2. Twelve official mechanism probes

### Probe conclusions

All twelve sections of `probe852.py` ran green and pinned down: the five window functions with `povey = hann^0.85`; blackman equalling `torch.blackman_window` at `0.42` but not `0.5`; both `snip_edges` framing formulas with reflect padding and the `(0,0)` too-short return; `_get_log_energy` with epsilon always `1.1920928955078125e-07` and the floor applied in the log domain; the replicate first sample of preemphasis; the energy timing of `raw_energy=True/False`; the full chain order from dither to post energy; the `1127/700` mel convention; `mel_freq_delta` divided by `num_bins+1` and the `(num_bins,1)` shape of `center_freqs`; the four-assignment VTLN overwrite order; all three mel columns being warped when warp is not 1; column 0 of `spectrogram` being overwritten by energy; and the switches and `htk_compat` tail orchestration of fbank / mfcc.

```text
WINDOW_TYPES = ['hamming', 'hanning', 'povey', 'rectangular', 'blackman']
```

## 3. Three-layer cross-check

### Official layer

`official_kaldi.py` executes the official `_feature_window_function` / `_get_strided` / `_get_window` / `get_mel_banks` / `vtln_warp_freq` / `fbank` / `mfcc` verbatim with float64 tensors.

### Reference layer

The problem's reference `solution.py`; all five modes go through `solve(mode, **kw)` in pure Python float64.

### Oracle layer

`verify852.py` rewrites the formulas independently in pure Python without importing any function from the reference: a naive `cmath.exp` DFT, mel and inverse mel, the triangular filterbank, the three-piece VTLN, the preprocessing chain, the DCT and the lifter are all implemented separately.

### Cross-check conclusion

The five modes produce 22556 check items, final result ALL GREEN. official-vs-reference uses `tol=1e-6, rel=1e-5` by default and `tol=5e-5, rel=0.0` for mfcc `features` alone; oracle-vs-reference always uses `1e-7`. A `pad-negative` case was added to cover the `pad <= 0` branch.

## 4. Official quirk list

### The center_freqs shape

The official docstring claims `center_freqs` is `(num_bins,)`; during the smoke test `round()` on a list raised `TypeError: type list doesn't define __round__ method`, and the actual return is `(num_bins, 1)`. The reference keeps the nested list.

### float32 quantization error

Inside `get_mel_banks`, `torch.arange(int64) * float` is demoted to float32, giving `center_freqs` about `1e-3` of error at a few thousand Hz, which accumulates through the DCT into roughly `2.5e-5` absolute error on mfcc `features`. The first cross-check round had 241 failures, all in `OFF mb*.center_freqs`; a relative tolerance cut them to 57, and a separate relaxation for mfcc finally cleared them.

```text
chk_off(tag, got, exp, tol=1e-6, rel=1e-5)
chk_off("mf%d.features" % ci, ref["features"], off, tol=5e-5, rel=0.0)
```

### VTLN overwrite order

The official comment says `order of operations matter here`. The four assignments `after_h -> before_h -> before_l -> outside_low_high_freq` rely on overlapping regions, and out-of-range frequencies are kept unchanged.

### blackman coefficient

At the default `blackman_coeff=0.42`, the official hand-written formula equals `torch.blackman_window` exactly; the first probe had the assertion reversed and was corrected after measurement to "`0.42` equal, `0.5` unequal".

## 5. Implementation points of the five modes

### vtln_warp

Compute `l/h/scale/Fl/Fh/scale_left/scale_right` first, then write `res` in the four-assignment order; `_warp_mel_py` takes inverse mel of a column, warps, and wraps back with mel.

### mel_banks

`mel_freq_delta` uses the denominator `num_bins+1`; warp=1 uses `max(0, min(up,down))` and warp!=1 uses the mask branch; `center_freqs` stays `(num_bins,1)`; `high_freq<=0` and `vtln_high<0` each add the nyquist once.

### framing

`_get_strided` has two framing formulas with reflect padding; `_get_window_py` runs the preprocessing chain in a fixed order with `raw_energy` deciding the energy timing; `_FRAMING_DEFAULTS` supplies the defaults; `padded_window_size` must be even and no smaller than the window length.

### fbank

Right zero column on `bins`, per-frame `_rfft_abs`, squaring under `use_power`, log under `use_log_fbank`, energy column position decided by `htk_compat` under `use_energy`, and per-column CMS under `subtract_mean`.

### mfcc

Calls fbank forced to `True, True, use_energy, htk_compat, False`; force-overwrites the DCT first column; C0 of the lifter is always `1.0` and the lifter is skipped at `0.0`; `htk_compat` moves C0/energy to the tail.

```text
dct[:, 0] = sqrt(1 / num_mel_bins)
```

## 6. Domain constraints and error messages

Every invalid input raises `ValueError` with a message naming the violated constraint: `Invalid window type` for the window type, `must be positive` for `window_shift`, `must be >= window_size` and `must be even` for `padded_window_size`, `must be provided` and `shape mismatch` for dither, `vtln_low must be higher` / `vtln_high must be lower` / `must be positive` / `inflection points out of range` for VTLN, `at least 3 mel bins` / `must be even` / `Bad values in options: low-freq / high-freq / nyquist` / `vtln-low / vtln-high` for `mel_banks`, `num_ceps cannot be larger than num_mel_bins` for mfcc, and `unknown mode` for an unrecognized mode.

```text
Bad values in options: vtln-low / vtln-high
```

## 7. Test case design

### Independent oracle

Each of the 7 `check_*(solve)` cases carries a pure-Python oracle rewritten from the official formulas and compares every returned field; waveforms and noise come from a deterministic LCG and depend on no randomness source. The test cases must not `import solution`, and every `check_*` must accept a `solve` argument.

### Two-way assertions

Besides matching the oracle, the result must differ from a plausible wrong candidate: the HTK mel convention, the `num_bins` denominator, `max/min` versus the mask branch, zero padding versus reflect padding, energy-domain clamp versus log domain, `warp=1` versus `warp!=1` bins, the head/tail `use_energy` column, the C0 overwrite, and the `htk_compat` tail orchestration.

### Domain constraint cases

Invalid inputs must still have legal shapes so that deleting the target check is not masked by downstream shape validation; the message keyword is asserted as well (`vtln-low` rather than the broad `vtln`). An all-zero waveform pins the default `energy_floor=1.0`, and an explicit blackman branch pins the default `blackman_coeff=0.42`.

## 8. Mutation coverage

`mut852.py` contains 154 original mutants covering mel / epsilon / log-energy, five window functions, `_get_strided` framing, the `_get_window` preprocessing chain, the four VTLN assignments and return fields, every `get_mel_banks` arithmetic site and both branches, DFT / CMS, fbank switches, DCT / lifter / mfcc orchestration, nine defaults, and the `solve` dispatch. The first round gave 146 CAUGHT / 7 SURVIVED / 1 ANCHOR-ERROR: 4 were confirmed truly equivalent (`vt-before-l-cond`, `mb-mask-up-cond`, `mb-mask-down-cond`, `dct-ortho-zero`) and registered as excluded, 3 test gaps were filled, and 1 anchor was fixed; after wiring, the rerun gave **150 effective mutants: 150 CAUGHT / 0 SURVIVED / 0 ERROR**.

## 9. Common pitfalls

- The judging environment has no `torchaudio`, so a literal official execution must inject a fake `create_dct`.
- The `center_freqs` quantization error between official float32 and reference float64 needs a relative tolerance, with a separate relaxation for mfcc.
- `center_freqs` is `(num_bins,1)`; do not `round()` a nested list in a helper.
- The oracle DFT with `math.fsum` over complex values raises `TypeError: must be real number, not complex`; use a naive `cmath.exp` accumulation.
- `cepstral_lifter=0.0` must skip the lifter or it divides by zero.
- Test cases use `check_*(solve)` and the mutation runner likewise uses `getattr(T, name)(S.solve)`.
- Asserting only the exception type for domain constraints is insufficient; the message keyword must be asserted.
- Equivalent mutants should keep their record but be excluded from the statistics, and anchors must be unique.

## 10. Self-check list

- Three-layer cross-check: 22556 items ALL GREEN.
- Test cases: 7/7 pass against the reference.
- Mutation: 150 effective mutants all CAUGHT.
- Bilingual validation: zero new errors / warnings.
- selftest: all ACCEPTED in both run and submit modes.
