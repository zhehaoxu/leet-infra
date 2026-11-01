# FastSpeech2 duration chain and variance adaptor ledger

## Problem

FastSpeech2 (official implementation `ming024/FastSpeech2`, `model/modules.py`) runs a
VarianceAdaptor between encoder output and decoder input. With predictor outputs given
as explicit inputs, reproduce the **per-operator ledger** of everything that happens
after them:

1. **Duration chain** (when `duration_target is None`):
   `duration_rounded = clamp(round(exp(logd) - 1) * d_control, min=0)`
   - `exp(logd)-1` (FastSpeech1 legacy offset) -> `round` (**banker's rounding**:
     2.5->2, 3.5->4, 0.5->0) -> multiply by `d_control` -> clamp min=0. Order is fixed.
   - A fractional `d_control` makes the product fractional (2·0.4=0.8); the official
     `LengthRegulator.expand` uses `max(int(size), 0)` and **truncates** — 0.8 -> 0
     frames, so speeding up speech can swallow short phonemes.
2. **Training branch**: `duration_target` given -> pass through verbatim (no rounding,
   no d_control).
3. **Expansion ledger**: each phoneme's feature repeats `max(int(d), 0)` times and is
   concatenated; zero duration emits no frames; `expand_indices` records the source
   phoneme of every frame; mel_len = total frames.
4. **Bucket index**: `torch.bucketize` defaults to `right=False` = `bisect_left`:
   **value exactly equal to bins[k] falls into bucket k** (100->0, 200->1, 300->2).
5. **log-bins float endpoint quirk**: bins = `exp(linspace(log min, log max, n_bins-1))`
   has rounded endpoints (min=50 gives bins[0]=49.999...<50), so **v = the nominal min
   50 falls into bucket 1**; in linear mode endpoints are exact and v=min falls into
   bucket 0. The two modes differ at the boundary.
6. **p_control multiplies before bucketize**: 150×1.5=225 vs 150×2.0=300 (exactly on an
   upper boundary -> bucket 2) must reproduce the boundary-crossing effect.

## Interface

```python
solve(log_durations, duration_target, d_control,
      expand_features, feat_value, p_control,
      feat_min, feat_max, n_bins, log_scale)
```

Returns a dict:

```python
{
  "duration_rounded": list,   # final per-phoneme durations (possibly fractional floats)
  "mel_len": int,             # total expanded frames
  "expand_indices": list,     # source phoneme index of each frame
  "feat_bucket": int,         # bucketize(feat_value * p_control, bins)
}
```

Constraints: `len(log_durations) == len(expand_features)`; when `duration_target` is
given its length must equal the phoneme count and elements are non-negative ints;
`n_bins >= 2`, `feat_min < feat_max` (n_bins=1 gives empty bins, bucket always 0).

## Example

```python
solve([math.log(3), math.log(1)], None, 1.0, [0.5, -0.5],
      150.0, 1.0, 100.0, 300.0, 4, False)
# durations: round(e^log3 - 1)=2, round(e^0 - 1)=0
# feat: 150 in [100,200) -> bucket 1
# -> {"duration_rounded":[2,0], "mel_len":2, "expand_indices":[0,0], "feat_bucket":1}

solve([math.log(3), math.log(4)], None, 0.4, [1.0, 1.0],
      0.0, 1.0, 1.0, 2.0, 2, False)
# round(2)*0.4=0.8 -> truncated to 0 frames; round(3)*0.4=1.2 -> 1 frame
# -> {"duration_rounded":[0.8,1.2], "mel_len":1, "expand_indices":[1], "feat_bucket":1}

solve([math.log(1)], None, 1.0, [1.0], 50.0, 1.0, 50.0, 800.0, 256, True)
# log-bin endpoint 49.99...<50 -> v=50 falls into bucket 1 (float quirk)
# -> {"duration_rounded":[0], "mel_len":0, "expand_indices":[], "feat_bucket":1}
```

## Scale and judging

- Phoneme count T in [0, 8]; n_bins in [1, 256]; 200 random cross-checks (oracle uses
  bisect_left and an independent expansion path).
- Mutation coverage: round->int truncation, dropping the -1 offset, missing clamp,
  bisect_left->right, p_control times->plus, log bins without exp, linspace endpoint
  shift, target branch consuming d_control — 13 mutations all caught.
- Intermediate values follow float64 semantics; `round` must be banker's rounding
  (Python built-in and torch.round agree).
