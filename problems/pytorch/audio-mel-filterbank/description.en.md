# Audio Front-End: Mel Filterbank and Log-Mel

## Problem

Human hearing is not linear in frequency: roughly linear below 1 kHz and roughly logarithmic above. The Mel scale models that perception, and the **Mel filterbank** is the matrix that collapses a linear power spectrum into `n_mels` bands. `log-mel` is what Whisper / HuBERT actually consume as input features.

What bites in practice is not the triangle shape but the two scale definitions:

- **HTK scale**: `mel = 2595 · log10(1 + f/700)`. This is the HTK book and early Kaldi convention, logarithmic everywhere.
- **Slaney scale**: linear below 1 kHz (`mel = f / (200/3)`, so 1000 Hz maps to exactly mel = 15), logarithmic above. librosa's default.

The two nearly coincide around 1 kHz but diverge badly at both ends. A separate switch is **area normalisation**: multiply each filter by `2 / (hz[m+2] - hz[m])` so all bands carry comparable units (otherwise wide high-frequency bands integrate to larger values for free). That is librosa's `norm="slaney"`; we bind it to the Slaney scale and treat the pair as one convention.

## Task

Implement `solve(power_spec, sample_rate, n_mels, f_min, f_max, scale)`:

- Inputs: `power_spec` `(T, F)` linear power spectrum (`F` bins uniformly spread over `[0, sr/2]`); `n_mels`; `f_min` / `f_max`; `scale ∈ {"htk", "slaney"}`.
- Take `n_mels + 2` equally spaced edges in the mel domain (including `f_min`, `f_max`) and map them back to Hz as `hz[0..n_mels+1]`.
- Triangle filter `m` rises and falls over `[hz[m], hz[m+1], hz[m+2]]` and is zero elsewhere.
- For `scale == "slaney"` multiply by `2 / (hz[m+2] - hz[m])`; `htk` keeps no normalisation.
- `mel = power_spec @ filters.T`, `logmel = log(max(mel, 1e-10))`.
- Outputs `(filters (n_mels, F), mel (T, n_mels), logmel (T, n_mels))`, all float32.

## Example

```
sr = 8000, n_mels = 3, f_min = 0, f_max = 4000, scale = "htk"
bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
mel edges (Hz) = [0, 428.0, 1119.0, 2232.2, 4000]

filters =
[[0, 0.893459, 0.165692, 0,        0,        0,        0,        0,        0      ]
 [0, 0.106541, 0.834308, 0.650824, 0.198715, 0,        0,        0,        0      ]
 [0, 0,        0,        0.349176, 0.801285, 0.842585, 0.561724, 0.280862, 0      ]]

power_spec[0] = [1..9]
mel[0]    = [2.283994, 6.312877, 16.637600]
logmel[0] = [0.825926, 1.842592, 2.811665]
```

## Constraints

- **The bindings are fixed**: Slaney scale plus area normalisation is one convention, HTK scale without normalisation is the other; do not cross them.
- **Edges are equally spaced in the mel domain**, not in Hz — that is the entire point of a Mel filterbank. Equal Hz spacing degenerates to a plain filterbank.
- **The `1e-10` floor**: silence has `mel = 0`, so `logmel = log(1e-10) ≈ -23.03`, never `-inf` (which would poison training).
- **Strictly zero outside the band**: with `f_min > 0` the low columns are all zero; likewise on the high side when `f_max < sr/2`.
- **A DC bin sits on the left edge of the first filter with weight 0**: a pure DC signal is therefore ignored by the whole bank — an easy detail to miss.
- **Invalid inputs raise `ValueError`**: unknown `scale`, not 2-D, `n_mels < 1`, `f_min < 0`, `f_min >= f_max`, `f_max > sr/2`, fewer than 2 bins.
