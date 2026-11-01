# GLM-4-Voice interleaved speech-text stream parsing ledger

## Problem

The GLM-4-Voice assistant output (zai-org/GLM-4-Voice `web_demo.py`) is an interleaved
stream of text tokens and audio codec tokens (`<|audio_N|>`); the demo parses it token
by token and streams chunks of progressively increasing size into the TTS. Reproduce
the parse loop verbatim (order-sensitive; official quirks kept):

Per token, in this exact order:
1. `token_id == end_token_id` -> `is_finalize = True`;
2. **the flush check precedes classification of the current token**:
   `len(audio_tokens) >= block_size` or (`is_finalize` and non-empty audio) -> emit a
   chunk (contents, current block_size, finalize flag); advance `block_size_idx` if not
   at the end of the ladder; clear audio_tokens. Note: end with empty audio does NOT
   flush;
3. classification only when `not is_finalize` — **after the end token every remaining
   token stops being classified** (complete_tokens stops growing); audio value =
   `token_id - audio_offset` (no upper-bound check officially).

There is **no final flush** after the stream ends: leftover sub-block audio is silently
dropped (official quirk).

## Interface

```python
solve(token_ids, audio_offset, end_token_id, block_size_list)
```

Returns:

```python
{
  "complete_tokens": list,   # every token id classified before finalize
  "text_tokens": list,       # all non-audio token ids
  "chunks": list,            # per chunk [audio_values, block_size_at_flush, is_finalize]
  "dropped_tail": list,      # audio values left unflushed after the stream
  "final_block_size": int,   # block size at stream end (after upgrades)
  "is_finalize": bool,
}
```

Constraints: block_size_list non-empty and strictly increasing;
audio_offset > max(text token id).

## Example

```python
solve([100, 200, 201, 202, 300], 200, 300, [25, 50, 100])
# -> {"complete_tokens":[100,200,201,202], "text_tokens":[100],
#     "chunks":[[[0,1,2],25,True]], "dropped_tail":[], "final_block_size":25,
#     "is_finalize":True}

solve(list(range(200, 230)), 200, 999, [25, 50, 100])
# at the 26th audio the check sees len=25>=25 (check precedes classification)
# -> chunks=[[[0..24],25,False]], dropped_tail=[25,26,27,28,29],
#    final_block_size=50, is_finalize=False
```

## Scale and judging

- Stream length <= 530; 400 random cross-checks; 14/14 mutations caught (check order,
  `>=`->`>`, dropping the offset subtraction, ignoring the end token, unconditional
  finalize flush, ladder upgrade behavior, missing clear, classifying after end,
  keeping the tail, constant flags/size, etc.).
