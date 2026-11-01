def solve(prompt_hiddens, prompt_embeds, audio_mask,
          thinker_hiddens, thinker_embeds,
          text_bos_embed, eos_embed, pad_embed,
          codec_embeds, codec_tokens, n_decode):
    L = len(prompt_hiddens)
    assert len(prompt_embeds) == L == len(audio_mask)
    T = len(thinker_hiddens)
    assert len(thinker_embeds) == T and T >= 1
    assert n_decode <= T + 1

    prefill = []
    for i in range(L):
        e = 0.0 if audio_mask[i] else prompt_embeds[i]
        prefill.append(prompt_hiddens[i] + e)
    reply = [thinker_hiddens[i] + thinker_embeds[i] for i in range(T)]
    prefill.append(text_bos_embed)
    prefill.append(reply[0])

    prefill_input_ids = [-1] * L + [-2, -3]

    queue = reply[1:] + [eos_embed, pad_embed]
    initial_reply_queue = list(queue)

    steps = []
    for t in range(n_decode):
        summed = codec_embeds[codec_tokens[t]] + queue[0]
        steps.append([summed, t])
        queue = queue[1:]

    return {
        "prefill_embeds": prefill,
        "prefill_input_ids": prefill_input_ids,
        "initial_reply_queue": initial_reply_queue,
        "steps": steps,
        "final_reply_len": len(queue),
    }
