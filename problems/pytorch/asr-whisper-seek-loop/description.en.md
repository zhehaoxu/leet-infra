# Whisper 30s window seek loop ledger

## Problem

The main loop of openai/whisper `whisper/transcribe.py` (word_timestamps=False): 30s
sliding windows; segmentation and seek advancement are fully determined by timestamp
tokens. Decoding is stripped — each window's token sequence and no_speech/avg_logprob
diagnostics are inputs.

Constants: N_FRAMES=3000 (30s mel frames), time_precision=0.02s/token (HOP 160 / SR
16000), timestamp_begin=50364, input_stride=2, and **time_offset = seek x HOP/SR =
seek x 0.01** (mel-frame precision — distinct from the token precision 0.02; the dual
precision is the subtlest ledger point).

Per iteration (official order):
1. `segment_size = min(3000, content_frames - seek)`;
2. `no_speech_prob > 0.6` **AND** `avg_logprob < -1.0` -> drop the segment, seek +=
   segment_size;
3. `ts = tokens >= 50364`; `single_timestamp_ending = ts[-2:] == [False, True]`;
4. `consecutive` = adjacent timestamp-pair indices + 1 (**only adjacent pairs split** —
   alternating timestamps do not); if non-empty, slice segments (start/end = first/last
   token of the slice minus begin, times 0.02, plus offset); single ending -> seek +=
   segment_size; else `seek += (tokens[last_slice-1] - begin) x input_stride` (the
   unfinished tail segment is dropped);
5. no adjacent pair -> duration = (last ts - begin) x 0.02 if a trailing timestamp
   (≠ begin) exists, else the whole window (segment_size x 0.01); seek += segment_size.

## Interface

```python
solve(content_frames, windows, no_speech_threshold=0.6, logprob_threshold=-1.0)
# windows: [{"tokens": [...], "no_speech_prob": f, "avg_logprob": f}] (loop order)
```

Returns:

```python
{"segments": [(start, end)], "seeks": [...], "final_seek": int}
```

## Example

```python
solve(6000, [
  {"tokens": [50364, 100, 50364+500, 50364+1000], "no_speech_prob": 0.1, "avg_logprob": -0.5},
  {"tokens": [50364, 300, 50364+1500], "no_speech_prob": 0.1, "avg_logprob": -0.5},
  {"tokens": [50364, 9, 50364+1500], "no_speech_prob": 0.1, "avg_logprob": -0.5},
])
# adjacent pair at index 2,3 -> segment1 (0,10); unfinished tail dropped -> seek += 500*2 = 1000
# window2 offset=10s single ending -> (10,40) seek=4000; window3 -> (40,70) seek=6000
```

## Scale and judging

- Window count guaranteed by input; 300 random cross-checks; 13/13 mutations caught
  (min clipping, no_speech AND->OR / single gate, `>=`->`>`, single check, consecutive
  indices, AND->OR, segment end using first token, unfinished seek without stride,
  else trailing-ts gate, duration/offset precision 0.01<->0.02).
