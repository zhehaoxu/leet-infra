TS_BEGIN = 50364
N_FRAMES = 3000
INPUT_STRIDE = 2
TIME_PRECISION = 0.02


def solve(content_frames, windows, no_speech_threshold=0.6, logprob_threshold=-1.0):
    segments = []
    seeks = []
    seek = 0
    widx = 0
    while seek < content_frames:
        w = windows[widx]
        widx += 1
        segment_size = min(N_FRAMES, content_frames - seek)
        time_offset = seek * 0.01  # HOP_LENGTH/SAMPLE_RATE: mel 帧精度
        tokens = w["tokens"]
        if (w["no_speech_prob"] > no_speech_threshold
                and w["avg_logprob"] < logprob_threshold):
            seeks.append(seek + segment_size)
            seek += segment_size
            continue
        ts = [t >= TS_BEGIN for t in tokens]
        single_ending = ts[-2:] == [False, True]
        consecutive = [i + 1 for i in range(len(ts) - 1) if ts[i] and ts[i + 1]]
        if consecutive:
            slices = list(consecutive)
            if single_ending:
                slices.append(len(tokens))
            last_slice = 0
            for cur in slices:
                st = tokens[last_slice:cur]
                start = time_offset + (st[0] - TS_BEGIN) * TIME_PRECISION
                end = time_offset + (st[-1] - TS_BEGIN) * TIME_PRECISION
                segments.append((round(start, 3), round(end, 3)))
                last_slice = cur
            if single_ending:
                seeks.append(seek + segment_size)
                seek += segment_size
            else:
                last_pos = tokens[last_slice - 1] - TS_BEGIN
                seeks.append(seek + last_pos * INPUT_STRIDE)
                seek += last_pos * INPUT_STRIDE
        else:
            duration = segment_size * 0.01
            tss = [t for t in tokens if t >= TS_BEGIN]
            if tss and tss[-1] != TS_BEGIN:
                duration = (tss[-1] - TS_BEGIN) * TIME_PRECISION
            segments.append((round(time_offset, 3), round(time_offset + duration, 3)))
            seeks.append(seek + segment_size)
            seek += segment_size
    return {
        "segments": segments,
        "seeks": seeks,
        "final_seek": seek,
    }
