# torchaudio NCCF pitch-detection ledger: fourth-power normalization in _compute_nccf, half-window preference in _combine_max, left replicate padding in _median_smoothing, and the float32 quirk of the frequency conversion

## 1. Background

`torchaudio.functional.detect_pitch_frequency` estimates pitch frame by frame with a normalized cross-correlation function (NCCF). This problem splits the whole chain into five independently judged pure-Python numerical ledgers: framed cross-correlation, threshold combination, half-window maxima, median smoothing, and the final frequency conversion.
This problem belongs to a different algorithm family from `audio-yin-trough-parabolic-ledger` (838, librosa YIN: cumulative mean normalized difference, parabolic interpolation and a trough threshold). Here we implement torchaudio NCCF cross-correlation, half-window preference and one-sided median smoothing.

### Official source

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  _compute_nccf          (lines ~1014-1058)
  _combine_max           (lines ~1061-1069)
  _find_max_per_frame    (lines ~1072-1099)
  _median_smoothing      (lines ~1102-1118)
  detect_pitch_frequency (lines ~1121-1163)
```

### Verbatim execution environment

The staging file `.workbuddy/scratch/torchaudio858/official858.py` uses `ast.get_source_segment` to extract the five top-level functions byte-for-byte from the 2511-line official file already downloaded for problem 855, and `exec`s them into a shared namespace. Only `math`, `typing`, `torch`, `Tensor` and `Tuple` are injected; no function body is mocked. The driver layer uses float64 throughout.

## 2. Conventions

### Inputs and outputs

```text
solve(mode, **kw) -> dict
mode = nccf | combine_max | find_max | median_smooth | pitch
```

The reference solution exposes the single entry point `def solve(mode, **kw)`. The judge file holds a set of top-level `def check_*(solve)` functions into which the judge injects `solve`. Judge cases must not use `from solution import ...`.
All inputs are nested Python lists. The implementation is pure Python: no `torch`, `torchaudio` or `numpy`, and no global RNG. Any leading dimensions are flattened in C order into equal-length rows, computed row by row and folded back with `lead + [tail]`.

### Numeric precision and invalid inputs

Judged numerics use float64. The official last line, `freq = sample_rate / (EPSILON + indices.to(torch.float))`, casts the integer index to float32, which swallows the `1e-9` epsilon entirely: index 4 gives exactly 200.0. The judged path recomputes the same formula in float64 and obtains `199.99999995`; `freq_official` exposes the verbatim official behaviour separately.
Every invalid input raises a `ValueError` with a specific keyword: `waveform must be a non-empty numeric array`, `waveform must be an array`, `sample_rate must be an integer`, `sample_rate must be at least 1`, `frame_time must be a positive number`, `freq_low must be an integer`, `freq_low must be at least 1`, `freq_high must be at least 1`, `win_length must be at least 3`, `nccf must have at least 2 dimensions`, `nccf last dimension is too short for freq_high`, `a and b must have the same length`, `thresh must be a positive number`, `indices length is too short for win_length`, or `unknown mode`. Validation order is part of the contract.

## 3. Mode 1: nccf

### Framing and right zero padding

```text
lags       = ceil(sample_rate / freq_low)
frame_size = ceil(sample_rate * frame_time)
num_frames = ceil(waveform_length / frame_size)
pad        = lags + num_frames * frame_size - waveform_length
padded     = waveform + [0.0] * pad
s1 = padded[i * frame_size : i * frame_size + frame_size]
s2 = padded[lag + i * frame_size : lag + i * frame_size + frame_size]
```

`lag` runs from 1 to `lags` inclusive, giving `lags` columns; `i` runs from 0 to `num_frames - 1`, giving `num_frames` rows. The output shape is `lead + [num_frames, lags]`. Padding is on the right only, so the tail frame sees zeros.

### Fourth-power normalization (contrary to the docstring)

```text
EPSILON = 1e-9
dot = sum(s1[k] * s2[k] for k in range(frame_size))
nr1 = sqrt(sum(a * a for a in s1))
nr2 = sqrt(sum(b * b for b in s2))
phi = dot / (EPSILON + nr1)**2 / (EPSILON + nr2)**2
```

The official docstring states `dot / sqrt(E1 * E2)`, but the source performs two successive divisions by two squared norms (a fourth-power normalizer), so values are far below 1. The `1e-9` epsilon is visible as `0.2499999995 != 0.25`. This normalizer is NOT scale invariant: multiplying the whole waveform by 2 divides every value by 4 (plus an epsilon drift), although `argmax` and hence the final frequency are unchanged.

## 4. Mode 2: combine_max

`_combine_max(a, b, thresh=0.99)` computes `mask = a_values > thresh * b_values` elementwise; on a hit it takes BOTH the value and the index from `a`, otherwise from `b`. The comparison is strict: `a == 0.99 * b` does not hit. `thresh` is an explicit input of this mode, defaults to `0.99`, and must be positive. All four arrays must be non-empty numeric arrays of equal length.

## 5. Mode 3: find_max

`lag_min = ceil(sample_rate / freq_high)`; `best = max(nccf[..., lag_min:])`; `half_size = lags // 2`; `half = max(nccf[..., lag_min:half_size])`. The official code calls `_combine_max(half, best)` (note `a` is the first half-window and the threshold is HARDCODED to 0.99) to prefer a smaller lag, then applies two offsets: `indices += lag_min` and `indices += 1`.
`argmax` returns the first maximum to match `torch.max`. When `half_size <= lag_min` the official `torch.max` fails on an empty tensor, so the reference rejects it early with `nccf last dimension is too short for freq_high`. This mode requires `nccf` to have at least two dimensions.

## 6. Mode 4: median_smooth

`pad_length = (win_length - 1) // 2` is applied as LEFT replicate padding only (no right padding); `unfold(-1, win_length, 1)` produces the sliding windows and `torch.median` takes the LOWER median `sorted[(n-1)//2]` for even windows.
The output has `n + pad_length - win_length + 1 = n - ceil((win_length - 1) / 2)` frames, i.e. fewer than the input. For `win_length = 1 / 2`, `pad_length = 0` and the official `torch.cat([])` raises, so the reference narrows the domain to `win_length >= 3`.

## 7. Mode 5: pitch

The batch is first packed into `[-1, L]` rows, then nccf, find_max and median_smooth run in order; finally `freq = sample_rate / (1e-9 + index)` and the result is unpacked to the input leading dimensions. The intermediate ledgers `nccf`, `indices` and `smoothed` are returned as well, reshaped to non-packed form according to the input rank.
Defaults are `frame_time = 0.01`, `win_length = 30`, `freq_low = 85` and `freq_high = 3400`. `meta` carries `lags`, `frame_size`, `num_frames`, `pad` and `padded_length`.

## 8. Returned fields

```text
nccf          : nccf shape waveform_length lags frame_size num_frames pad padded_length
combine_max   : values indices mask thresh
find_max      : indices lag_min half_size best_values best_indices half_values half_indices shape
median_smooth : smoothed padded roll pad_length shape
pitch         : freq shape nccf indices smoothed meta
```

## 9. Self-test and judging

The 12-segment official probe passes 57/57; the three-layer cross-check (verbatim official, reference solution, independent definitional oracle) passes 471/471; there are 21 top-level `check_*(solve)` cases and the reference passes 21/21; mutation testing starts with 202 mutants, removes 11 true equivalents and catches all remaining 191.
Every expected value was pinned against the verbatim official environment; no tolerance is used to hide a mechanism difference. `scripts/check_bilingual.py` enforces bilingual consistency across the whole library.
