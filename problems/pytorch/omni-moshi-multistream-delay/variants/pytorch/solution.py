UNGENERATED = -1


def solve(max_steps, input_frames, dep_outputs):
    assert max_steps >= 1
    gen, inp, vocab, delay = 8, 8, 2049, 2
    apad = vocab - 1
    total = gen + inp
    audio_tokens = [[UNGENERATED] * total for _ in range(max_steps + delay)]
    playable = []
    for step in range(max_steps):
        frame_in = input_frames[step] if step < len(input_frames) else []
        for c, t in enumerate(frame_in):
            audio_tokens[step][gen + c] = t
        out = dep_outputs[step] if step < len(dep_outputs) else None
        for c in range(gen):
            d = 0 if c == 0 else delay
            pos = max(step - d, 0)
            audio_tokens[pos][c] = apad if out is None else out[c]
        if step <= delay:
            playable.append(None)
        else:
            frame = audio_tokens[step - delay - 1]
            if any(v >= vocab - 1 for v in frame):
                playable.append(None)
            else:
                playable.append(list(frame))
    ung = sum(row.count(UNGENERATED) for row in audio_tokens)
    return {
        "audio_tokens": audio_tokens,
        "playable": playable,
        "ungenerated_left": ung,
    }
