# notes — asr-whisper-quality-gate (order 671)

## Source

- Origin: `openai/whisper`, `whisper/transcribe.py` (`decode_with_fallback` and the segment-level `should_skip`), `whisper/utils.py` `compression_ratio` (zlib, not the gzip module).
- The three official constants and semantics: `compression_ratio_threshold=2.4` (only `>` counts as too repetitive), `logprob_threshold=-1.0` (only `<` counts as too low), `no_speech_threshold=0.6`.
- Two error-prone points:
  - **silence override has the highest priority**: when `no_speech_prob > 0.6 and avg_logprob < -1.0`, even a previously set needs_fallback=True flips back to False ("confirmed silence, stop retrying"); the window then passes at temperature 0 with no fallback;
  - **the segment exemption has the opposite direction**: skip requires `no_speech_prob > 0.6` and **not** exempted by `avg_logprob > -1.0`; exact equality with a threshold triggers neither (strict inequalities).
- Boundary with #109 (TTS): this is a **post-decoding quality ledger**; boundary with candidate ⑨ VAD: VAD is front-end segmentation.

## Judging

- solve args: windows (texts/avg_logprob/no_speech_prob, one column per temperature), temperatures, three thresholds. Returns per-window ledgers.
- Oracle: `test_cases._py` is an independent implementation (a while loop advancing temperature by temperature, a separate covered-flag path), 120 random cross-checks (1–4 temperatures, half the texts made repetitive, boundary values 0.59/0.6/0.61 and -1.0 mixed in).
- Anchors: pass at temperature 0; high-cr fallback ("abc"*20 cr≈4.29); silence override (nsp=0.9/alp=-2 at every temperature → passes at temperature 0 but action=skip); high nsp with high alp exemption; all temperatures exhausted picks the last (officially no fail concept); the three exact-equality boundaries cr/alp/nsp; empty text cr<1 (zlib header overhead); multibyte counted by utf-8 bytes.
- Invalid inputs: empty windows/temperatures, missing/extra window keys, metric length ≠ temperature count, non-str texts, NaN/Inf/bool metrics, negative temperature, bool threshold → ValueError.

## Mutation log (9/9 caught)

cr-threshold-gte (`>` → `>=`), logprob-lte, silence-no-logprob-cond (override loses the logprob condition), no-silence-override (override deepens instead), skip-exempt-inverted (exemption inverted), skip-gte, no-break-on-accept (no break on pass, always the last temperature), cr-inverted (compression ratio inverted), exhausted-pick-first (exhaustion picks temperature 0).

## Red lines

No model calls or decoding (metrics all explicit inputs); no real audio IO; boundary with the `audio` front-end collection (signal processing): this is a **post-decoding quality ledger**.
