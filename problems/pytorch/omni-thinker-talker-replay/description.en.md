# Qwen2.5-Omni thinker-talker hidden-state replay ledger

## Problem

In transformers `modeling_qwen2_5_omni.py` (v4.52) the Talker replays every thinker
(text LLM) token hidden state frame by frame into the audio LM. With the dimension
compressed to D=1, reproduce the full data-flow ledger (networks stripped; hidden
states and embeddings are explicit scalar inputs):

1. **Prefill**: `talker_embed[i] = prompt_hiddens[i] + prompt_embeds[i]`, except that
   where `audio_mask[i]` is True the embedding is zeroed first (**only the embedding —
   the hidden state survives**, official masked_scatter_ semantics); two frames
   `[text_bos_embed, reply[0]]` are appended at the tail, where
   `reply[i] = thinker_hiddens[i] + thinker_embeds[i]`.
2. **input_ids pattern**: `[codec_mask]*L + [codec_pad, codec_bos]` (encoded mask=-1,
   pad=-2, bos=-3).
3. **Reply queue**: initial = `reply[1:] + [eos_embed, pad_embed]`, length T+1 —
   eos/pad backstop the two decode steps after the text eos (official constraint
   n_decode <= T+1; an empty queue crashes).
4. **Decode 1:1**: each step `summed = codec_embeds[codec_tokens[t]] + queue[0]`, then
   the head is dropped. One codec token consumes exactly one thinker frame.

## Interface

```python
solve(prompt_hiddens, prompt_embeds, audio_mask,
      thinker_hiddens, thinker_embeds,
      text_bos_embed, eos_embed, pad_embed,
      codec_embeds, codec_tokens, n_decode)
```

Returns:

```python
{
  "prefill_embeds": list,       # length L+2
  "prefill_input_ids": list,    # [-1]*L + [-2, -3]
  "initial_reply_queue": list,  # reply[1:] + [eos, pad]
  "steps": list,                # per step [summed, consumed_idx]
  "final_reply_len": int,       # T + 1 - n_decode
}
```

Constraints: the three L-length inputs match; T >= 1; `n_decode <= T + 1`;
codec_embeds covers all used tokens.

## Example

```python
solve([1.0, 2.0], [10.0, 20.0], [False, True],
      [100.0, 200.0], [1.0, 2.0],
      0.5, 0.1, 0.2, {5: 0.05, 6: 0.06}, [5, 6, 5], 3)
# prefill: audio embedding zeroed -> [11.0, 2.0], tail [0.5, 101.0]
# -> prefill_embeds=[11.0, 2.0, 0.5, 101.0], prefill_input_ids=[-1,-1,-2,-3]
#    initial_reply_queue=[202.0, 0.1, 0.2]
#    steps=[[202.05, 0], [0.16, 1], [0.25, 2]], final_reply_len=0
```

## Scale and judging

- L, T <= 6; 300 random cross-checks; 14/14 mutations caught (add semantics, audio
  embedding-only zeroing, tail frames and their order, bailout frames in the queue,
  decode head-drop, ids encoding, the reply addition formula, etc.).
- Float anchors compared with `math.isclose`.
