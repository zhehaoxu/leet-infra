# FastSpeech2 duration chain and variance adaptor ledger

## Background and sources

FastSpeech2 (2306.lei; official implementation `ming024/FastSpeech2`,
`model/modules.py`) is the workhorse TTS acoustic model. Between encoder and decoder
its VarianceAdaptor performs three deterministic steps: durations -> frame expansion,
and pitch/energy quantized-bucket embedding lookup. The predictors are convnets, but
**everything after their output is pure scalar/index arithmetic** — exactly the scope
of this problem. Note the official `LengthRegulator.expand` is naive per-phoneme
repetition (`vec.expand(max(int(size), 0))`); there is no "remainder distribution" —
that phrasing belongs to third-party write-ups, and this problem follows the official
source.

## Mechanism ledger

1. **Duration chain** (inference): `duration_rounded = clamp(round(exp(logd) - 1) * d_control, min=0)`.
   Order matters: `exp(logd)-1` first (a FastSpeech1 +1-offset legacy), then torch.round
   (**banker's rounding**: round(2.5)=2, round(3.5)=4, round(0.5)=0), then multiply by
   d_control, then clamp min=0.
2. **Non-integer duration truncation**: with d_control != 1 the product can be fractional
   (round(2)·0.4 = 0.8); the official `expand` does `max(int(size), 0)`, which **truncates** —
   0.8 -> 0 frames, so a fast speech rate can erase short phonemes entirely.
3. **Training branch**: when `duration_target` is given, duration_rounded = target
   **verbatim** (no rounding, no control).
4. **Expansion ledger**: each phoneme's feature repeats `max(int(d), 0)` times and is
   concatenated; mel_len = total frames; zero-duration phonemes emit nothing.
5. **Bucket index**: `torch.bucketize(value, bins)` defaults to `right=False` = `bisect_left`:
   **value exactly equal to bins[k] falls into bucket k** (100->0, 200->1, 300->2).
6. **log-bins float endpoint quirk**: `exp(linspace(log min, log max, n_bins-1))` rounds
   the endpoints (min=50 gives bins[0]=49.999...<50), so **v = nominal min 50 falls into
   bucket 1**; in linear mode the linspace endpoints are exact and v=min falls into bucket 0.
7. **Control multiplier position**: p_control multiplies **before** bucketize
   (`prediction * control` then lookup); after ×1.5 a value can land exactly on an upper
   boundary and jump buckets.

## Judging notes

- 12 fixed anchors + 200 random cross-checks (oracle uses bisect_left and an independent
  expansion path).
- 13 mutations all caught: round->int truncation, dropping the -1 offset, missing clamp,
  bisect_left->right, p_control times->plus, log bins without exp, linspace endpoint shift, etc.
  `max(int(d), -1)` is a mathematically equivalent mutation (the duration chain already
  guarantees non-negativity) and is not recorded as a required catch.
- linspace must be **endpoint-exact** (torch.linspace semantics): `lo + i*(hi-lo)/(num-1)`
  deviates from true linspace in midpoints at the float level. The oracle uses the same
  formula so cross-checks are unaffected, but the log-endpoint anchor (v=50 -> bucket 1)
  depends on the actual rounding of `exp(linspace(log·))`.
