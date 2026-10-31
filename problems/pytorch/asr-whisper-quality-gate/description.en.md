# Whisper Quality Gate and Temperature Fallback Ledger

## Problem

Whisper's `transcribe` decodes every 30s window with a temperature fallback sequence and judges window quality with three thresholds: zlib compression ratio (repetition loops), average log probability (untrustworthy), and no_speech probability (silence). This problem transcribes that **window-level fallback + segment-level skip** as a pure ledger: each window carries per-temperature decoding metrics (texts / avg_logprob / no_speech_prob), and the output is the compression-ratio ledger, selected temperature, fallback chain, silence coverage, and the final action.

## Solution

The ledger is pinned by the spec (item-by-item identical to the official `decode_with_fallback` plus segment-level `should_skip`; constants 2.4 / -1.0 / 0.6):

1. `compression_ratio(text) = len(utf8 bytes) / len(zlib.compress(utf8 bytes))` (official `whisper/utils.py`);
2. **Per-temperature fallback** (independent test per temperature, order is priority):
   - `needs_fallback = cr > cr_t` (too repetitive) **or** `avg_logprob < lp_t` (too low);
   - if `no_speech_prob > ns_t` **and** `avg_logprob < lp_t` → `needs_fallback = False` (silence override, highest priority);
   - `not needs_fallback` → select this temperature and stop; all temperatures exhausted → select the last one (official behavior: return the final result);
3. **Segment-level skip** (using the selected temperature's metrics):
   - `should_skip = no_speech_prob > ns_t`;
   - if `avg_logprob > lp_t` (**strictly greater**) → exempt, do not skip;
   - a skipped window is fast-forwarded (official `seek += segment_size` semantics), otherwise `transcribe`.

Note the two comparison directions are opposite: the fallback's logprob condition is `<`, the segment exemption is `>`; equality with a threshold triggers neither.

Returns per window `{"compression_ratios", "selected_temp_index", "fallback_chain", "silence_covered", "action"}`.

## Constraints

- 1 ≤ len(windows), 1 ≤ len(temperatures) ≤ 10, each metric list length equals the temperature count.
- Empty windows, missing/extra window keys, ragged metric lengths, non-str texts, non-numeric or NaN/Inf metrics, negative temperatures, bool/non-numeric thresholds → `ValueError`.
