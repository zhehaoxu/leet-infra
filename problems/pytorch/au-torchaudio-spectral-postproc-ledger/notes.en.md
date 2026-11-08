# 855 solution notes: torchaudio spectral post-processing ledger

## 1. Official source retrieval

### Source list and dependencies

The main source is `pytorch/audio main::src/torchaudio/functional/functional.py` (2511 lines / 95067 bytes). Unlike Captum, torchaudio still lives under the `pytorch` owner and has not moved to `meta-pytorch`; fetching the base64 content from `api.github.com/repos/pytorch/audio/contents/...?ref=main` works. The judge environment has no torchaudio, so `import torchaudio.functional` is impossible; the official file also has a relative `.filtering` import at the top, so executing the whole file directly fails.

### Verbatim execution environment

The solution mirrors the `create_dct` extraction from problem 852: `ast.get_source_segment` pulls the verbatim source of the seven target functions, which are then exec-ed into a shared `NS` after injecting `torch` / `math` / `warnings` / `Tensor` / typing names.

```text
WANT = ["spectrogram", "_get_spec_norms", "_get_complex_dtype",
        "amplitude_to_DB", "DB_to_amplitude",
        "compute_deltas", "sliding_window_cmn"]
NS = {"math": math, "warnings": warnings, "torch": torch,
      "Tensor": torch.Tensor, "Optional": ..., "Union": ...}
for name in WANT:
    seg = ast.get_source_segment(SRC, node)
    exec(compile(seg, "functional.py::" + name, "exec"), NS)
```

The driver converts nested lists into `float64` tensors. `run_spectrogram` temporarily replaces `torch.stft` with a spy that records the exact kwargs and restores it in a `finally` block; complex results are always split into `{"real": ..., "imag": ...}`. The global default dtype is untouched, because `float64` inputs already make `stft` produce `cdouble`.

## 2. Twelve official mechanism probes

### Probe conclusions

All 87 assertions across twelve probes pass. Highlights: the four valid inputs and two exception kinds of `_get_spec_norms`; the 10/20 multipliers, `amin` clamp, `db_multiplier` subtraction, 4D per-batch cutoff and 3D/2D single cutoff of `amplitude_to_DB`; the `power=1/0.5` round trips of `DB_to_amplitude`; the `n`, `denom`, cross-correlation sign, replicate/constant boundary difference and 3D pack/unpack of `compute_deltas`; the CMN window schedule, equivalence to a naive window mean, `norm_vars`, the `window_frames==1` zeroing and 2D rank reduction; the `_get_complex_dtype` mapping; the stft kwargs, `pad` constant padding, `return_complex` warning and normalization guards of `spectrogram`; plus a manual DFT comparison covering frame construction, centered window zero padding, center reflect, both `frame_length` and `window` scaling, the power branches and `onesided=False`.

## 3. Three-layer comparison

### official layer

The verbatim official functions, covering dB on 2D/3D/4D/5D inputs, inverse dB on 1D/2D inputs, deltas with W=3/4/5/7, seven CMN parameter combinations, thirteen spectrogram variants and 2D waveforms.

### reference layer

The five modes of `solution_staged.py`, all pure Python with no torch dependency; `_shape` / `_flatten` / `_reshape` handle flattening and reshaping for arbitrary dimensions.

### oracle layer

An independent implementation: dB uses an explicit per-batch `log10`; deltas uses `sum_i i*(c[t+i]-c[t-i]) / (2*sum i^2)`; CMN uses a naive window mean and variance; spectrogram uses a DFT built from `cmath.exp(-2*pi*i*k*j/n_fft)`. The three layers are structurally different, preventing shared mistakes.

### Comparison conclusion

```text
total checks: 1520    failures: 0
RESULT: ALL GREEN
```

Only two checks failed on the first run, caused by writing the string alias of `torch.cfloat` as `torch.complex64` in an assertion; comparing dtype objects directly fixed it. A separate probe failure came from reading the official output layout as `[time][freq]` instead of `[freq][time]`; after fixing the indices, all twelve probes passed.

## 4. Official quirk checklist

### CMN slice typo

The official first-frame slice `input_part = specgram[:, window_start : window_end - window_start, :]` should use `window_end` as its upper bound. But this branch runs only when `last_window_start == -1`, i.e. at `t=0`, and the first frame's `window_start` is clamped to 0 regardless of centering, so `window_end - window_start == window_end`: the typo is **unobservable**. The probe enumerated centered/non-centered modes, many `cmn_window` values and frame counts; the first-frame start is always 0. The reference keeps the verbatim form (`range(ws, we - ws)`), and the corresponding mutant is registered as truly equivalent.

### 2D squeeze rank reduction

The trailing `if len(input_shape) == 2: cmn_specgram = cmn_specgram.squeeze(0)` reduces a `(1, F)` output to `(F,)`. Thus a 2D input with `num_frames == 1` produces a 1D list rather than a 2D one; 3D inputs are unaffected.

### docstring dimension contradiction

The `sliding_window_cmn` docstring describes the input as `(..., time, freq)` and the output as `(..., freq, time)`, yet the source uses `num_frames, num_feats = input_shape[-2:]` and finally reshapes back to `input_shape[:-2] + (num_frames, num_feats)`, i.e. exactly the input shape.

## 5. Implementation points of the five modes

### db

Compute `vals` first, then split into chunks of `packed_channels * freq * time`, derive one cutoff per batch and apply elementwise `max`. Key detail: `num_batches = prod(shape[:-3])` exceeds 1 only when the rank is greater than 3; for 3D/2D it is always 1. `cutoffs` must be `None` when `top_db` is `None`.

### db_inverse

Map `ref * (10^(0.1*v))^power` elementwise, allowing 1D input; the result echoes `ref` and `power`. Paired with `db`, the two combinations 10/power=1 and 20/power=0.5 round-trip.

### deltas

Flatten all leading dimensions into channels, apply replicate/constant padding per channel, then compute the `kernel[j]*padded[t+j]` cross-correlation. Even `win_length` must use `(W-1)//2`; `denom` uses the official closed form; the `kernel` metadata is `[-n..n]`.

### cmn

Compute the full `windows` list first, then maintain `cur_sum` / `cur_sumsq` with a one-shot accumulation on the first frame and incremental remove/add using `last_window_start` / `last_window_end` afterwards. The `norm_vars` branch multiplies the row by `variance^-0.5`, and `window_frames==1` zeroes the row directly. The 2D single-frame squeeze is reproduced at the end.

### spectrogram

The order is `pad` -> `center` padding -> centered window zero padding -> per-frame windowing -> DFT -> `frame_norm` scaling -> `window_norm` scaling -> power branch. `window_norm` uses the original window's squared sum; `frame_norm` divides by `sqrt(n_fft)`. Complex output is `[freq][time]`, and a 2D waveform adds a leading channel dimension.

## 6. Domain constraints and error messages

Every invalid input raises `ValueError` with a message naming the constraint: `amin must be a positive number`, `top_db must be a number or None`, `at least two dimensions`, `must be rectangular`, `must contain only numbers`, `Window length should be greater than or equal to 3`, `win_length must be an int`, `mode must be replicate or constant`, `cmn_window must be a positive int`, `norm_vars must be a bool`, `waveform must be 1D or 2D`, `window length must equal win_length`, `Invalid normalized parameter`, `Input type not supported` (`TypeError`), `pad_mode must be reflect or constant`, `window energy must be positive`, `reflect pad requires`, `waveform too short for n_fft`, and `unknown mode`.

## 7. Judge case design

Eight `check_*(solve)` functions: `check_db` (4D/5D per-batch cutoffs and `packed_channels>1`), `check_db_inverse` (round trips), `check_deltas` (even W, sign, padding difference, 3D), `check_cmn` (seven parameter sets, centered vs non-centered difference, 2D single-frame reduction, norm_vars), `check_spectrogram` (thirteen variants, 2D waveform, `pad` making bins differ from frames), `check_domain` (about 35 exact-message assertions), `check_metadata` (metadata field by field) and `check_unknown_mode`. The driver mimics the judge injecting `solve`; the reference scores 8/8.

## 8. Mutation coverage

```text
202 raw mutants, 5 true equivalents removed, 197 active: 197 CAUGHT, 0 SURVIVED, 0 ERROR
ALL MUTANTS CAUGHT
```

The first run gave 168 CAUGHT / 27 SURVIVED / 6 ANCHOR-ERROR. The SURVIVED mutants were mostly deleted domain guards with no matching invalid input, fixed by adding about twenty exact-message assertions. Three further cases were unobservable and needed dedicated test data: 4D with `packed_channels>1` (all earlier cases had exactly one channel), an even `win_length` (both roundings agree for odd W), and a 2D waveform with `pad=2` so that bins differ from frames. The six ANCHOR-ERRORs were duplicate anchors (`ws = t - cmn_window` is a prefix of `ws = t - cmn_window // 2`; an 8-space indentation matches a 12-space line; two `center` guards; identical accumulation lines in the first-frame and incremental branches); all were replaced with longer unique anchors.

Five truly equivalent mutants: `cmn-slice` (first-frame start is always 0), `h-reshape-scalar` (the scalar branch is unreachable in this input domain), `h-rows-cast` (`float(x)` does not change numeric values), `sp-cos-sign` (cos is even), and `sp-power1` (`abs(z)^1 == abs(z)`).

## 9. Common pitfalls

- The official complex output is `[freq][time]`; comparing it as `[time][freq]` shifts everything.
- The reflect left segment starts at `x[p]` and the right segment at `x[-2]`; using `x[p-1]` / `x[-1]` only breaks the first frame and is easy to miss.
- The CMN `min_cmn_window` condition is `window_end > t`, not "window frame count below the minimum".
- Normalization must test `str` before `bool`, otherwise `True` (bool is an int subclass) takes the wrong branch.
- The deltas `pad_mode` collides with the outer `solve(mode, ...)`, so `solve` must rename and map it.

## 10. Self-check list

Twelve official probes green, 1520 three-layer comparisons ALL GREEN, driver 8/8, mutation 197/197 CAUGHT, and only then the four bilingual documents, meta, starter precheck, eleven landed files, dual-mode selftest, full-library bilingual check, collections registration, todos update, short guard and full guard.
