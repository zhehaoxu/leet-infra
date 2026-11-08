# Notes on the official mechanism and its pitfalls

## Official source and version

- Repository: `librosa/librosa`, default branch `main`.
- Files used: `librosa/core/pitch.py` (`_cumulative_mean_normalized_difference`, `_parabolic_interpolation`, `_pi_stencil`, `yin`) and `librosa/util/utils.py` (`localmin`, `_localmin_sten`, `tiny`).
- How it was fetched: the GitHub API, `git/trees/main?recursive=1` plus `contents?ref=main` plus `base64.b64decode`.
- The judging environment (`.venv-eval`) has **neither** librosa nor numba, so the official cross-check script expands the numba stencils line by line into equivalent plain numpy/Python, staying strictly aligned with the source; `tiny` is taken from `np.finfo(np.float64).tiny`.

## Four traps in the CMND stage

1. **The `E[0]` zeroing quirk**: the official code first takes the `cumsum` of the squared signal, then immediately sets `yin_frames[..., 0, :] = 0`, and only then writes `d[k]`. So `d[1] = 2*(r0 - r1) - 0`. Without the zeroing, `d[1]` would wrongly subtract `sq[0]` and every later `cummean` would be shifted.
2. **The `- E[k-1]` index**: the difference subtracts the **energy prefix sum**, not the difference itself, and the index is `k-1`. numpy evaluates the right-hand side into a temporary array before writing back, so `d[k]` still sees the original cumsum values.
3. **The divisor of `cummean` is `k_range = [1, 2, ..., max_period]`**, i.e. the number of differences, not `tau`. Using `tau` breaks everything when `min_period != 1`.
4. **The `den` slice offset is `min_period - 1`**: `den = cumulative_mean[min_period-1 : max_period]`, with length `max_period - min_period + 1`. This is the easiest place to lose a `-1`.

In addition, `tiny` is added to the **denominator** (`yin_numerator / (yin_denominator + tiny)`). For normal inputs it barely affects the numbers, but without it an all-zero `den` would divide by zero, so this piece must be judged too.

## Two traps in the parabolic interpolation stage

1. **`|b| >= |a|` is a suppression that includes equality.** The official stencil writes `np.abs(b) >= np.abs(a)` and returns 0 on equality. Relaxing it to a strict `>` yields a wrong nonzero shift when `|b| == |a|` (for instance `x = [-0.5, 0.0, 1.5]`, where `a = b = 1.0`). When `a = 0`, the suppression branch always wins because `|b| >= 0` is always true, so there is no division by zero.
2. **The edge bins are explicitly zeroed.** After the stencil, `_parabolic_interpolation` executes `shiftsi[..., -1] = 0` and `shiftsi[..., 0] = 0`. The default boundary mode of the numba stencil already emits zeros, so these two lines are redundant but explicit; the task still treats the edges as constantly zero, and the test set includes cases where `yin_period` lands on the first or last bin (there `shift` must be 0 and `period` must equal `min_period + idx`).

## Four traps in the trough picking stage

1. **Strict on the left, non-strict on the right**: the condition of `util.localmin` is `x[i] < x[i-1]` and `x[i] <= x[i+1]`. This is the opposite of `localmax`, and the two sides cannot be swapped. The judging grid must **mix in tied values** (quantizing random values onto a fixed scale), otherwise the difference between `<` and `<=` is essentially unmeasurable on purely random floats.
2. **The last element is special-cased**: it has no right neighbor, so the official code uses `xi[..., -1] < xi[..., -2]`, strict on both sides.
3. **Bin 0 is overridden by the official code** to `x[0] < x[1]` (the stencil boundary would otherwise be constantly False).
4. **The threshold is strictly less-than**: `x < trough_threshold`, so a value equal to the threshold does not count as a trough. `argmax` takes the **first** True, `argmin` also takes the first index on ties; when everything is False `argmax` returns 0, but it must be overridden by `global_min`.

Finally `period = min_period + yin_period + shift`, and `f0 = sr / period`. Note that `min_period` is the start of the tau axis and must not be dropped.

## How it was verified and how the test cases were designed

- Official cross-check: 600 groups for each of the three modes, **all 1800 groups consistent**.
- Test cases: one `check_*` per mode, each with an independent plain-Python oracle, several hand-computed anchors, and a grid comparison.
- The hand-computed anchors deliberately pin down: the `E[0]` quirk, the `-E[k-1]` index, the `k_range` divisor of `cummean`, the `min_period-1` offset of `den`, `tiny` on the denominator, the equal-boundary case of `|b| >= |a|`, the forced zero shift on the edges, the strict-left / non-strict-right asymmetry of `localmin`, the doubly strict last element, the override of column 0, the strict `x < threshold`, `argmax` taking the first True, `argmin` taking the first index on ties, the fallback to the global minimum when everything is False, `min_period + idx`, and `f0 = sr / period`.
- Mutation testing: 40 single-point mutants, all of which must be **CAUGHT**. The lessons from this round are worth recording:
  - The first run had 39 mutants and 6 survivors. Diagnosis showed two were **truly equivalent mutants** (the `shifts` list is already initialized to 0.0, so deleting the explicit "zero the edges" assignment does not change the result). Per the established principle they must be dropped; rewriting them into the form "let the edges take part in the parabola computation" makes them genuinely testable.
  - The other three survivors (left strict vs non-strict and the strictness of the last element in `localmin`) came from a grid of purely continuous random floats with no tied values. Mixing the grid into four styles (continuous, narrow range, quantized to 1/3, binary 0/1) fixed that.
  - One survivor (`>=` vs `>`) needed an exact equality case `|b| == |a|`, which the `x = [-0.5, 0.0, 1.5]` anchor supplies.
- **The lesson from 836 still holds: a fully green mutation run cannot replace a self-check of the correct solution.** After every change to the test cases this round, the `run`/`submit` double-mode self-check of the correct solution was executed first and confirmed `ACCEPTED` before running the mutations, so as to avoid the situation where "the correct solution was actually wrong and the mutants were falsely declared dead".

## Boundaries kept out of the judging domain

FFT autocorrelation (`audio.autocorrelate`), the HMM/Viterbi decoding and random sampling inside `pyin`, the spectral peak estimation of `piptrack`, the exception branches of `__check_yin_params`, `center`/`pad_mode` framing, multi-channel broadcasting, real waveform IO and resampling are all kept out of the judging domain. The task keeps only the three deterministic pure numeric ledgers.
