# Moshi multistream acoustic delay and overwrite chain ledger

## Problem

kyutai-labs/moshi (rust `lm_generate_multistream.rs`, Config::v0_1) runs 16 Mimi audio
codebooks per step — the left half of 8 is the generated stream (depformer), the right
half of 8 the input stream (user speech). Model sampling is stripped (depformer outputs
are inputs); only the token matrix and playback ledger are examined.

Constants: generated=8, input=8, vocab=2049 (audio_pad=2048=vocab-1), acoustic_delay=2.

Per step (source order):
1. Write the input half: `audio_tokens[step][8+c] = input[c]` (short streams write
   nothing).
2. Write the generated half: `for c in 0..8`: delay = 0 if c==0 else 2; target slot
   `max(step-delay, 0)` (saturating_sub — steps 0..2 keep overwriting slot 0 with
   cb1-7, an officially commented behavior); None output writes audio_pad. **cb8 is not
   in the write loop** (it is only a read-side zero-delay codebook).
3. Playback `last_audio_tokens()`: `step <= 2` -> None; else read frame
   `audio_tokens[step-3]`; any value >= 2048 in the frame -> None (whole frame muted),
   otherwise the 16 values are playable.

Matrix size `(max_steps + 2) x 16`, initialized UNGENERATED (-1).

## Interface

```python
solve(max_steps, input_frames, dep_outputs)
```

Returns:

```python
{
  "audio_tokens": list,    # full matrix (incl. trailing delay rows)
  "playable": list,        # per step playable frame list or None
  "ungenerated_left": int, # remaining UNGENERATED cells
}
```

Constraints: max_steps >= 1; input values < 2048; dep outputs < 2048 or None.

## Example

```python
solve(6, [[100+i for i in range(8)]]*6,
      [[10+step*8+c for c in range(8)] for step in range(6)])
# slot0 = [10, 27..33] (cb0 = step0 output, cb1-7 overwritten by step2)
# slot1 = [18, 35..41]; slot2 = [26, 43..49]
# playable[0..2]=None; playable[3]=slot0+input frame (playable)
```

## Scale and judging

- max_steps <= 10; 300 random cross-checks; 13/13 mutations caught (cb0 delay flip,
  saturating removal, write loop including cb8, None->0, input-half drop/zeroing,
  playback window bounds, frame offset, any->all, threshold >, matrix tail).
  `list(frame)`->`frame` is a mathematically equivalent mutation (no later write hits a
  slot after it is recorded; reference and copy are indistinguishable) — not recorded.
