# 858 Solution Notes: torchaudio NCCF Pitch-Detection Ledger

## 1. Official sourcing

### Source files and dependencies

```text
.workbuddy/scratch/captum855/functional.py   2511 lines / 95067 bytes
.workbuddy/scratch/torchaudio858/official858.py   ast.get_source_segment x5
WANT = [_compute_nccf, _combine_max, _find_max_per_frame,
       _median_smoothing, detect_pitch_frequency]
NS injects only: math typing torch Tensor Tuple Optional List
```

### Verbatim execution environment

The five functions are `exec`d into a shared `NS`. The driver exposes `comp_nccf`, `comp_combine_max`, `comp_find_max`, `comp_median_smooth` and `comp_pitch`, all in float64. `comp_pitch` runs the verbatim `detect_pitch_frequency` first to save `freq_official`, then recomputes the frequency in float64 from the same intermediates and reshapes `nccf`, `indices` and `smoothed` from packed `[B, ...]` back to the input leading shape.

## 2. Twelve-segment official mechanism probe

### Probe conclusions

P1-P4 pin the framing meta, the pad identity, the fourth-power normalizer and tail-frame padding. P5 pins the strict boundary of `mask = a > thresh * b`. P6-P8 pin `lag_min`, half-window preference and the two offsets. P9-P11 pin left replicate padding, unfold windows, the lower median for even windows and frame shrinkage. P12 pins the full frequency ledger and batch unpack.
Three assertions were wrong on the first run: P8's double offset is `0 + 2 + 1 = 3`, P10's `w=30` output is 25 frames rather than 26, and P12's frequency is exactly 200.0 because of float32. All were corrected against the official run. P10 also records the `torch.cat([])` crash for `win_length=1`. After `comp_pitch` was switched to float64 the probe was rerun, with P12 pinning both `199.99999995` and `freq_official = 200.0`.

## 3. Three-layer cross-check

### Official layer

The five verbatim functions plus torch float64 tensors. The left replicate pad and `unfold` of `_median_smoothing` are recomputed separately so `padded` and `roll` can be reconciled.

### Reference layer

Pure-Python `solution_staged.py` with five modes. `pack` / `unflatten` handle arbitrary leading dimensions, `lower_median` uses `sorted[(n-1)//2]` and `argmax` returns the first maximum.

### Oracle layer

An independent definitional implementation: a single-row kernel with one `math.fsum` division, `max((v, -k))` to pick the first maximum, and independent `_rows` / `_rebuild` helpers. It reuses nothing from the reference solution.

### Cross-check result

```text
VERIFY 858 DONE checks=471 fails=0 ALL GREEN
coverage: nccf x6 / combine x5 / find x5 / median x6 / pitch x4
          includes 1D / 2D / 3D leading dimensions
```

## 4. Judge cases and hardening patch

### First-run red lights

Two of the first 12 judge cases were red. `check_median_smooth` wrongly expected 5 frames for `w=3, n=7` (the official result is 6 frames, `[9,8,2,7,3,6]`). `check_nccf` wrongly assumed NCCF is scale invariant (doubling the waveform actually divides values by 4). Both expectations were corrected against the verbatim official run; the reference solution always agreed with the official environment.

### Hardening patch

`patch858.py` adds nine `check_*` cases: non-integral ceil boundaries for lags / frame_size / num_frames, pointwise NCCF on an all-nonzero waveform, domain contracts for scalars / bools / missing keys / bool flags, validation of each of combine_max's four arrays, find_max with an odd lag dimension and the `0.99` / `0.98` threshold boundaries, find_max's own lower bounds, non-numeric median_smooth input, pitch guard ordering, and pass-through of `freq_high` at a small sample rate. Final result: 21/21 ACCEPTED.

## 5. Mutation testing

### True equivalents

Eleven mutants were judged truly equivalent: (1) both error branches of `pack` and the `shape_of([])` branch are shadowed by the earlier `require_numbers` (4); (2) an `unflatten` leaf slice always has exactly one element (1); (3) the `-1e300` sentinel in `argmax` is below every judged finite value, verified with an all-negative row (1); (4) `sum(a*b)` equals `sum(b*a)` (1); (5) dropping one padded zero changes neither the dot product nor either norm (1); (6) swapping the order of the two squared-norm divisions, with a 400-case deterministic sweep giving `max |diff| = 6.94e-18` (1); (7) find_max indices are always integers, so `int(v) == float(v)` (1); (8) the `freq_high` lower bound in `_pitch` has exactly the same message as the one in `_find_max` (1).

### Final result

```text
PROBE  858 DONE ok=57 fail=0 ALL GREEN
VERIFY 858 DONE checks=471 fails=0 ALL GREEN
DRIVER 858 21/21 ALL ACCEPTED
MUT    858 202 total / 11 equivalent / 191 active / 191 CAUGHT
       0 SURVIVED, 0 ERROR, ALL MUTANTS CAUGHT
```

The first mutation round left 37 SURVIVED. All were closed by hardened judge cases or by equivalence rulings. In the process, `find_max` was found to expose a `thresh` parameter that does not exist in the official API; to match the hardcoded `_combine_max(half, best)` inside `_find_max_per_frame` exactly, that parameter was removed from the reference solution and the related mutation anchors were rewritten.

## 6. Official quirk list

(1) The docstring says `dot / sqrt(E1*E2)` but the source divides twice by squared norms. (2) The normalizer is not scale invariant (times 2 becomes divide by 4). (3) `_combine_max` uses strict `>` and its `0.99` threshold is hardcoded inside `_find_max_per_frame`. (4) Two index calibrations: `+= lag_min` and `+= 1`. (5) `_median_smoothing` pads only on the left, so the output is shorter than the input. (6) Even windows use the lower median. (7) `win_length=1/2` triggers the `torch.cat([])` crash. (8) The float32 cast in the frequency conversion swallows the `1e-9` epsilon. (9) Output length is `n - ceil((w-1)/2)`.

## 7. Deduplication and reuse

Before landing, a full scan confirmed `max_order = 857`, that 858 is free, that all 835 orders are unique and that the slug is new. A library-wide search of `meta.json` for `nccf`, `pitch_frequency` and `griffinlim` returned no hits. The boundary with 838 (librosa YIN) is the algorithm family; 855 (spectral post-processing) and 857 (waveform operators) share the same `functional.py` source but use disjoint function sets.
