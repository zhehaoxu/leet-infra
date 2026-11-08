# librosa YIN: the CMND cumulative mean normalized difference, parabolic interpolation, and trough threshold ledger

## Background

YIN (de Cheveigné & Kawahara, 2002) replaces autocorrelation with a "cumulative mean normalized difference" (CMND) function to estimate the fundamental frequency. librosa's `yin()` splits this into three independently verifiable numeric ledgers:

1. `_cumulative_mean_normalized_difference`: turn autocorrelation plus squared energy into the normalized difference `d'(tau)`;
2. `_parabolic_interpolation`: refine a discrete trough with a piecewise parabola to get sub-bin precision;
3. `util.localmin` plus an absolute threshold: pick the first trough below the threshold, fall back to the global minimum, then convert to period and frequency.

This task rebuilds only these three **pure numeric** ledgers. FFT autocorrelation, the HMM/Viterbi decoding inside `pyin`, `piptrack`, and real waveform IO are all out of scope — every input is a list of already-computed numbers.

## General conventions

- All inputs are lists of Python `float`; no tensors, no gradients, no batching.
- Floating point comparisons use absolute tolerance `1e-9` plus relative tolerance `1e-9`.
- `tiny` is the smallest positive normalized number of `numpy.float64`, `2.2250738585072014e-308`, which is what official `util.tiny()` returns.
- Every mode returns a `dict`; the field names are listed per mode below.
- Indices are zero-based and **inclusive** on both ends.

## Modes and rules

### 1. `cmnd`: cumulative mean normalized difference

Inputs:

| Parameter | Meaning |
|---|---|
| `sq` | the squared signal `y_m^2` of one frame, length at least `max_period + 1` |
| `acf` | autocorrelation `r[0..max_period]`, length exactly `max_period + 1` |
| `min_period` | minimum period, `1 <= min_period <= max_period` |
| `max_period` | maximum period |
| `tiny` | denominator guard, default `2.2250738585072014e-308` |

Steps:

```
E[k]      = cumsum(sq)[k]           # prefix sum of squared energy
E[0]      = 0                       # official quirk: E[0] is zeroed before d is written
d[0]      = 0
d[k]      = 2 * (acf[0] - acf[k]) - E[k - 1]       # k = 1 .. max_period
cummean[j] = (d[1] + ... + d[j+1]) / (j+1)        # j = 0 .. max_period-1
den[i]     = cummean[min_period - 1 + i]          # i = 0 .. max_period-min_period
cmnd[i]    = d[min_period + i] / (den[i] + tiny)
```

Returned fields:

- `difference`: length `max_period + 1`, i.e. `d[0..max_period]`;
- `cumulative_mean`: length `max_period`, i.e. `cummean[0..max_period-1]`;
- `denominator`: length `max_period - min_period + 1`, i.e. `den`;
- `cmnd`: length `max_period - min_period + 1`, the normalized difference itself.

Pitfalls:

- `E[0]` must be zeroed before computing `d`, otherwise `d[1]` wrongly subtracts `sq[0]`;
- `d[k]` subtracts `E[k-1]`, not `E[k]`;
- the divisor of `cummean` is `k_range = [1, 2, ..., max_period]`, not `tau`;
- the `den` slice starts at `min_period - 1`;
- `tiny` is added to the **denominator**, not the numerator.

### 2. `parabolic_shifts`: piecewise parabolic interpolation

Input `x`: a sequence of CMND values, length `n >= 3`.

For every interior index `i = 1 .. n-2`:

```
a[i]     = x[i+1] + x[i-1] - 2 * x[i]
b[i]     = (x[i+1] - x[i-1]) / 2
shift[i] = 0            if |b[i]| >= |a[i]|
         = -b[i] / a[i]  otherwise
```

The first and last bins are forced to zero: `shift[0] = shift[n-1] = 0`.

Returned fields: `a` (length `n`), `b` (length `n`), `shifts` (length `n`). The first and last entries of `a` and `b` are placeholder `0.0`.

Pitfalls:

- the suppression condition is `|b| >= |a|` (including equality), in which case the shift is 0; writing `>` yields a wrong nonzero shift when `|b| == |a|`;
- when `a = 0` the suppression branch always wins (`|b| >= 0` is always true), so no division by zero occurs;
- the edge bins are explicitly zeroed, not interpolated.

### 3. `trough_selection`: trough picking and the absolute threshold

Inputs:

| Parameter | Meaning |
|---|---|
| `x` | a sequence of CMND values, length `n >= 3` |
| `min_period` | minimum period |
| `sr` | sampling rate |
| `trough_threshold` | absolute threshold |

Steps:

```
is_trough[i] = (x[i] < x[i-1]) and (x[i] <= x[i+1])   # i = 1 .. n-2
is_trough[n-1] = x[n-1] < x[n-2]                      # the last element has no right neighbor
is_trough[0]   = x[0] < x[1]                          # official override for bin 0
is_threshold_trough[i] = is_trough[i] and (x[i] < trough_threshold)
global_min = argmin(x)                                # first index on ties
yin_period = argmax(is_threshold_trough)              # first True
if there is no True at all: yin_period = global_min
shift  = shifts[yin_period]
period = min_period + yin_period + shift
f0     = sr / period
```

Returned fields: `is_trough`, `is_threshold_trough` (both boolean lists of length `n`), `global_min`, `yin_period`, `no_trough_below_threshold` (boolean), `shift`, `period`, `f0`.

Pitfalls:

- the `util.localmin` condition is **strict on the left `<`, non-strict on the right `<=`**; swapping the two sides breaks it;
- the last element uses `x[n-1] < x[n-2]`, strict on both sides;
- bin 0 is explicitly overridden by the official code to `x[0] < x[1]`;
- the threshold comparison is `x[i] < trough_threshold`; **equal to the threshold does not count**;
- `argmax` takes the **first** True, and `argmin` also takes the first index on ties;
- when everything is False it must fall back to `global_min`;
- `period = min_period + yin_period + shift`, and `f0 = sr / period`.

## Input and output examples

`cmnd`:

```python
solve(
    "cmnd",
    sq=[1.0, 4.0, 9.0, 16.0, 25.0],
    acf=[55.0, 34.0, 20.0, 10.0, 4.0],
    min_period=2,
    max_period=4,
)
# {
#   "difference":      [0.0, 42.0, 65.0, 76.0, 72.0],
#   "cumulative_mean": [42.0, 53.5, 61.0, 63.75],
#   "denominator":     [53.5, 61.0, 63.75],
#   "cmnd":            [65.0/53.5, 76.0/61.0, 72.0/63.75],
# }
```

`parabolic_shifts`:

```python
solve("parabolic_shifts", x=[1.0, 0.2, 0.5])
# {
#   "a":      [0.0, 1.1, 0.0],
#   "b":      [0.0, -0.25, 0.0],
#   "shifts": [0.0, 0.25/1.1, 0.0],
# }
```

`trough_selection`:

```python
solve(
    "trough_selection",
    x=[0.5, 0.05, 0.3, 0.02, 0.4],
    min_period=2,
    sr=100.0,
    trough_threshold=0.1,
)
# {
#   "is_trough":                [False, True, False, True, False],
#   "is_threshold_trough":      [False, True, False, True, False],
#   "global_min":               3,
#   "yin_period":               1,
#   "no_trough_below_threshold": False,
#   "shift":                    0.1/0.7,
#   "period":                   3.0 + 0.1/0.7,
#   "f0":                       100.0/(3.0 + 0.1/0.7),
# }
```

## Out of scope

- `peak_pick` / `localmax` / greedy peak picking, and `util.localmin` variants other than the one used here;
- the FFT implementation of `audio.autocorrelate` and multi-frame broadcasting;
- the HMM/Viterbi decoding, random sampling and pitch smoothing inside `pyin`;
- the spectral peak estimation of `piptrack` and the exception branches of `__check_yin_params`;
- `center` / `pad_mode` framing and multi-channel handling;
- reading real waveform files and resampling.
