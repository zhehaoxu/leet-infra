# Qwen2.5-Omni thinker-talker hidden-state replay ledger

## Background and sources

The Talker in transformers `modeling_qwen2_5_omni.py` (v4.52) is the only mechanism of
its kind in the library: text-LLM hidden states are replayed frame by frame into an
audio LM. In the official `generate`, the thinker produces the whole text reply first;
its per-token last-layer hidden states and input embeddings are cached as two parallel
streams, and every Talker decode step consumes the head of that reply queue. This
problem strips the networks and compresses the dimension to D=1, keeping the full
data-flow ledger.

## Mechanism ledger

1. **Element-wise addition, not concatenation**: `talker_embed = thinker_hidden +
   token_embed` (both for every prefill position and every reply frame).
2. **Audio tokens zero only the embedding**: `embeds_to_talker` is `masked_scatter_`
   with zeros at audio/image/video placeholder positions — the embedding is gone but
   the hidden state survives (not a whole-frame drop).
3. **Prefill tail of two frames**: `[text_bos_embed, reply[0]]`; the matching input_ids
   pattern is `[codec_mask]*L + [codec_pad, codec_bos]`.
4. **Reply queue and bailout**: initial queue = `reply[1:] + [eos_embed, pad_embed]`,
   length T+1. The eos/pad frames backstop the two decode steps after the thinker's
   text eos — officially a reply may decode at most T+1 steps; an empty queue crashes
   (the official code has no guard).
5. **Decode is 1:1**: each step `summed = codec_embed(prev_token) + queue[0]`, then the
   queue head is dropped. One codec token consumes exactly one thinker frame — the
   actual "streaming throttle" ratio (1:1, no batched consumption).

## Judging notes

- 9 fixed anchors + 300 random cross-checks; 14/14 mutations caught (add->hidden-only,
  audio whole-frame drop / hidden zeroed, single tail frame, queue keeping reply[0],
  no bailout, eos/pad swap, decode without shift, ids pattern swap / all-mask, reply
  without embedding, final_len constant).
- Float anchors use `math.isclose` (binary representation of 0.01+0.1).
