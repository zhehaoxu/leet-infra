# 857 solution notes: torchaudio functional waveform-operations ledger

## 1. Official source

### Source files and dependencies

The main source is `pytorch/audio main::src/torchaudio/functional/functional.py` (2511 lines). Order 855 already fetched it through the GitHub API and stored it at `.workbuddy/scratch/captum855/functional.py`; 857 reuses that file instead of downloading again. Nine top-level functions are extracted: `mu_law_encoding` / `mu_law_decoding` (around lines 668-735), `_check_shape_compatible` / `_check_convolve_mode` / `_apply_convolve_mode` / `fftconvolve` / `convolve` (lines 2189-2315), `add_noise` (from line 2317) and `phase_vocoder` (lines 732-802). A full-bank scan confirmed order 857 was free and that no directory named `au-torchaudio-waveform-ops-ledger` existed; the `audio` collection had 8 problems, and the boundary against order 855 is waveform-side versus spectral-side, with no overlap.

### Verbatim execution harness

`official857.py` extracts each function whole with `ast.get_source_segment` (docstrings and comments included) and `exec`s it into a shared namespace `NS`. `NS` receives only `math`, `typing`, `warnings`, `torch` and a few typing aliases; no official function body is mocked. The driver converts nested lists to `float64` / `complex128` tensors:

```text
WANT = ["mu_law_encoding", "mu_law_decoding", "convolve", "fftconvolve",
        "_check_shape_compatible", "_check_convolve_mode",
        "_apply_convolve_mode", "add_noise", "phase_vocoder"]
for node in ast.parse(source).body:
    if isinstance(node, ast.FunctionDef) and node.name in WANT:
        exec(compile(ast.get_source_segment(source, node),
                     "functional.py::" + node.name, "exec"), NS)
```

The driver offers six comparison entry points: `comp_mu_law`, `comp_mu_law_decode`, `comp_convolve` (one entry for both direct and FFT), `comp_apply_mode`, `comp_add_noise` and `comp_phase_vocoder`. `comp_mu_law` explicitly re-wraps the int64 codes as a `float64` tensor before decoding, avoiding the official implicit `to(torch.float)` float32 demotion. `comp_add_noise` recomputes energies and the scale with the same official `vector_norm^2` formula so the intermediate ledger is observable.

## 2. Twelve official mechanism probes

### Probe findings

`probe857.py` runs 78 assertions in twelve segments, all green: P1 mu-law encoding endpoints / midpoint / int64 dtype / `+0.5` round-half-up / four `qc` values; P2 decoding formula, endpoints, tiny positive midpoint, monotonicity and 2D shape; P3 direct full-convolution values, a negative kernel, swapping when `x` is shorter without changing values; P4 `same` / `valid` using the **original** lengths with the convention unchanged by the swap; P5 center cropping for valid / same / full in `_apply_convolve_mode` and an invalid mode; P6 `fftconvolve` matching the direct method over four input pairs and three modes; P7 paired leading-dimension broadcast, FFT singleton leading broadcast, and the rank / non-broadcastable domain checks; P8 `add_noise` scalar snr, energies, 0 dB and 10 dB scales, `x + a*n`, snr rank and length mismatches; P9 per-row snr vectors, `lengths` masking only the energies while the padded region still receives noise, `lengths` rank and broadcasting; P10 `rate=1` returning the input verbatim (including the nested object and shape); P11 `time_steps` / `alphas` / frame counts for four rates matching `arange(0,T,rate)` and the ceil rule for `rate=1.3`; P12 phase accumulation, magnitude interpolation, keeping `phase_0`, tail magnitude decay from zero padding, and the 2D complex-spectrogram path.

## 3. Three-way verification

### Official layer

The nine verbatim function bodies cover four `qc` values for mu-law, six convolution inputs times three modes, four FFT cases, three slicing modes, scalar and vector snr with `lengths` for `add_noise`, and five rates plus batched leading dimensions for `phase_vocoder`, all in float64 / complex128.

### Reference layer

`solution_staged.py` has six modes in pure Python with no torch or numpy; `shape_of` / `flatten` / `unflatten` / `broadcast_shapes` / `broadcast_to` / `slice_last` / `map_vals` handle arbitrary-rank flattening, broadcasting and last-axis slicing. `_dft` / `_idft` are naive O(n^2) complex-exponential implementations reproducing the `rfft` / `irfft` product path; `cmath.phase` and explicit `cos` / `sin` stand in for the official `.angle()` and `torch.polar`.

### Oracle layer

An independent definitional implementation: convolution accumulates products in a triple loop per output position (never reusing `_conv_full`); the SNR scale is built directly as `sqrt(Es/En) * 10^(-snr/20)`; the phase vocoder expands explicit complex lists frame by frame with `cos` / `sin`; mu-law uses an independent `expm1` / `log1p` equivalent. All three layers agree on 961 checks.

### Verification result

```text
PROBE 857 DONE ok=78 fail=0 ALL GREEN
VERIFY 857 DONE checks=961 fails=0 ALL GREEN
DRIVER 857 12/12 ALL ACCEPTED   (before hardening)
DRIVER 857 20/20 ALL ACCEPTED   (after hardening)
```

## 4. Judge cases and the hardening patch

### First-round failures

Five of the initial 12 `check_*(solve)` cases failed. Source re-checking showed that **every failure was a wrong expected value in the judge case; the reference solution and the verbatim official harness agreed exactly**:

1. `check_mu_law` used `-0.4966767430305481` from the float32 demotion path; the official float64 value is `-0.4966766264665899` (the symmetric entry was corrected too). The `x=0.3` value `0.30633397981616217` was verified correct and kept.
2. `check_fftconvolve` listed `[0.15, 0.025, 0.275, 0.22, 0.08, 0.3]` for a mixed kernel; both the official result and hand convolution give `[0.15, 0.025, 0.225, 0.325, -0.025, 0.3]`.
3. `check_phase_vocoder` wrote 3 frames for the final `T=4, rate=1.3` case; the official `arange(0,4,1.3)` yields 4 frames with a last time step of `3.9`.
4. `check_mu_law_domain` put `1 / 0 / -4` in the non-integer group, where they hit `at least 2` first; split into a non-integer group `(2.5, "8", True)` and a lower-bound group `(1, 0, -4)`.
5. `check_phase_vocoder_domain` changed only `real` to 1D when testing insufficient rank, hitting `real and imag must have the same shape` first; `imag` was changed to 1D as well.

After correction, `driver857.py` reported 12/12 ALL ACCEPTED.

### Hardening patch

The first mutation round exposed 42 survivors, 18 of which were implemented behaviours that the judge could not observe. `patch857.py` appended eight groups of cases and two assertions: asymmetric three-dimensional leading broadcast (catching both reversed / final axis-order mutants), distinct last-dimension values under a singleton leading dim (catching broadcast indexing and build range), equal-length operands with no swap and a three-tap kernel (catching the `b[i-k]` negative-index coincidence on two-tap kernels and the `<=` swap), a positive center-crop start on 2D data (catching a dropped recursive start), `y_length=1`, validation order with two invalid inputs, a conflicting SNR leading dimension, and bool array elements; plus an FFT real-dtype assertion and numeric real/imaginary parts with the `freq` / `num_frames_in` ledger for the phase vocoder. Every added value was produced by the verbatim official harness first and then checked against the reference. The result was 20/20 ALL ACCEPTED.

## 5. Mutation testing

### True equivalents

`mut857.py` started with 201 mutants. Twenty-five were judged true equivalents and excluded, in four groups: (1) branches unreachable under the judged contract (`h-shape-empty`, `h-reshape-guard`, `cv-size-zero`, `pv-empty`, `ml-dec-range`, `ml-dec-guard`, etc., because entry-level `require_numbers` already rejects empty arrays and decoding is only applied to valid codes); (2) mathematical identities (`log1p` versus `log(1+x)` in `ml-log1p-inner`, `ml-dec-expm1`, `sm-valid-abs`, convolution commutativity in `cv-direct`, implicit DFT zero padding in `cv-fft-pada` / `cv-fft-padb`); (3) forms that are always identical under the current call shapes (`rows-reshape` with `numel([])==1`, `cv-lead-one` whose result is immediately cleared, `pv-rate1-ts` with `n==T`, `pv-pad` where one zero pad suffices, `h-bto-equal`, `h-bto-off`, `h-bto-rank`, `h-bto-scalar-value`, `h-unflatten-scalar`, `ml-dec-float`, `ml-x-guard`, `cv-size-msg`); (4) `pv-ph-wrap`: wrapping only adds an integer multiple of `2*pi` to the accumulated phase, leaving `cos` / `sin` unchanged; a separate deterministic sweep of 40 cases (5 rates times 8 phase-advance magnitudes) measured a maximum absolute difference of `2.187e-14`, far below the judge tolerance.

### Final result

```text
MUTANTS: 201 total, 25 true equivalents, 176 active
176 CAUGHT, 0 SURVIVED, 0 ERROR  --  ALL MUTANTS CAUGHT
```

Seven anchors appear twice in the source, so the mutation framework adds `ANCHOR_INDEX` (an occurrence index) to distinguish `_slice_mode` from `_slice_mode_only` and the same-named fields in two return dictionaries, preventing replacements at the wrong site.

## 6. Official quirks

1. `.to(torch.int64)` in `mu_law_encoding` truncates toward zero; the mapped value is non-negative, so `+0.5` is really round-half-up. `mu_law_decoding` demotes non-floating `x_mu` to float32 via `to(torch.float)`; the judged path avoids it with an explicit float64 re-wrap.
2. `convolve` swaps operands when the last dimension of `x` is shorter, yet `_apply_convolve_mode` still receives the original `x_size` / `y_size`; `fftconvolve` never swaps. Both implement true convolution via `conv1d(weight.flip(-1))` or `rfft`, not cross-correlation.
3. `_apply_convolve_mode` uses `max-min+1` for `valid` and always `x_length` for `same`; the start uses floor division, so an odd full length biases left.
4. `lengths` in `add_noise` masks only the energies; the addition still uses the unmasked `waveform` and `noise`, so the padded region receives scaled noise anyway.
5. `phase_vocoder` returns the input directly for `rate == 1.0`, skipping the polar round-trip; `time_steps = arange(0, T, rate)` gives `ceil(T/rate)` frames; the time padding is 2; phase is reduced by `phase_advance`, wrapped, added back, then shifted one frame by `cat([phase_0, phase[..., :-1]])` before `cumsum`.
6. Wrapping phase into `[-pi, pi]` changes it only by an integer multiple of `2*pi`, which is numerically equivalent for the final `cos` / `sin` (about `1e-14` floating difference); this was the only true equivalent requiring a numeric experiment to establish.

## 7. De-duplication and reuse

The `audio` collection grows from 8 to 9 problems. `au-torchaudio-spectral-postproc-ledger` (855) shares the same source file but covers only the spectral post-processing chain; this problem takes only the nine waveform-side functions, with disjoint mode sets, formulas and judged fields. The official `functional.py` stored by 855 is reused, as are the precheck / land / register script skeletons and the minimal-edit `collections.json` registration approach from 856. The `.venv-eval` judge environment has only torch and numpy; neither the reference solution nor the judge cases import any third-party library, and there is no global RNG.
