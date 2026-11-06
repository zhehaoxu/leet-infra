# Qwen3-ForcedAligner NAR Timestamp Ledger

## Background

Qwen3-ForcedAligner-0.6B (QwenLM/Qwen3-ASR) is an LALM-style **non-autoregressive** forced aligner: instead of CTC/DTW alignment it splits the transcript into words, inserts timestamp placeholders between and after words, and a single LLM forward's argmax is read only at those placeholder positions; multiplying by `timestamp_segment_time` yields a timestamp sequence that is post-processed into per-word (start, end) spans. This problem rebuilds that decidable post-processing chain (no model forward; argmax results are explicit inputs).

## Mechanism breakdown

### 1. Tokenization (tokenize)

`tokenize_space_lang`: split on whitespace → `clean_token` (keep only Unicode letters L\* / numbers N\* / apostrophe `'`, drop punctuation) → `split_segment_with_chinese` (CJK characters split one-by-one into single-char tokens, contiguous non-CJK runs merged), empty segments dropped. Both Chinese and English take this branch; Japanese/Korean go through external nagisa / soynlp tokenizers absent from the judge env, so they are excluded.

### 2. Placeholder encoding (encode)

`"<timestamp><timestamp>".join(word_list) + "<timestamp><timestamp>"`, prefixed with `<|audio_start|><|audio_pad|><|audio_end|>` — **each word has exactly 2 timestamp slots** (2 between words + 2 at the tail; even an empty word list keeps the tail pair).

### 3. NAR readout (masked_timestamps)

After `logits.argmax(-1)`, `output_ids[input_ids == timestamp_token_id]` takes predicted ids at placeholder positions, then multiplies by `timestamp_segment_time`. **Mask first, then multiply**: large values at non-placeholder positions never enter the result; the value count is exactly `2 × number of words`.

### 4. LIS monotonic repair (fix)

Timestamp predictions can be non-monotonic; the official code picks normal points via the longest **non-decreasing** subsequence (LIS):

```
dp[i] = 1; if data[j] <= data[i] and dp[j]+1 > dp[i] → dp[i]=dp[j]+1, parent[i]=j
max_idx = dp.index(max(dp))        # ties to the earliest
parent-chain backtrack → is_normal flags (the rest are anomalous)
```

Contiguous anomalous runs are handled by length:

- run length **<=2**: nearest normal values left/right (read from the already-repaired `result`), choose by distance `(k-(i-1)) <= (j-k)` (ties take left); when one side is missing, take the existing side;
- run length **>2**: `step = (right-left)/(anomaly_count+1)` linear interpolation (denominator includes both endpoints), or flat-fill when a side is missing;
- finally `[int(v) for v in result]` truncation (positive floats toward zero = floor).

### 5. Word-span mapping (parse)

After repair, each word takes `[2i]` as start and `[2i+1]` as end, then `round(v / 1000.0, 3)` converts to seconds (fixed is already int, so `/1000` has exactly 3 decimals; a wrong round precision is still observable).

## Judge domain

- Text, input_ids / output_ids, and timestamp sequences are all explicit inputs, fp64; mostly string and integer ledgers, no RNG;
- 5 modes: tokenize, encode, masked_timestamps, fix, parse;
- Excluded: LLM forward and real weights, Japanese/Korean external tokenizers, audio IO and resampling. Clearly delimited from `asr-whisperx-dtw-align` (CTC emission trellis DP backtracking) and `asr-cif-fire-emit` (emission state machine).
