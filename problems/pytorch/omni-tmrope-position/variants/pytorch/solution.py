"""TMRoPE：文本+音频段的 3D 位置 id 账（solution）。

转写 HF transformers qwen2_5_omni `get_rope_index`
（use_audio_in_video=False、无 image/video 分支）。
"""

TEXT = 0
AUDIO_START = 1
AUDIO_PAD = 2
AUDIO_END = 3


def audio_expand_len(mel_len: int) -> int:
    return ((mel_len - 1) // 2 + 1 - 2) // 2 + 1


def solve(input_ids, audio_seqlens):
    if not isinstance(input_ids, list) or not all(
        isinstance(t, int) and 0 <= t <= 3 for t in input_ids
    ):
        raise ValueError("input_ids must be a list of ints in [0, 3]")
    if not isinstance(audio_seqlens, (list, tuple)) or not all(
        isinstance(v, int) and v >= 1 for v in audio_seqlens
    ):
        raise ValueError("audio_seqlens must be ints >= 1")

    n_blocks = sum(1 for t in input_ids if t == AUDIO_START)
    if n_blocks != len(audio_seqlens):
        raise ValueError("audio block count must match audio_seqlens")

    # 结构校验：块必须为 [AUDIO_START, AUDIO_PAD, AUDIO_END]
    i = 0
    blocks = []  # (pad_index, mel_len)
    ai = 0
    while i < len(input_ids):
        if input_ids[i] == AUDIO_START:
            if i + 2 >= len(input_ids) or input_ids[i + 1] != AUDIO_PAD \
                    or input_ids[i + 2] != AUDIO_END:
                raise ValueError("malformed audio block")
            blocks.append((i, int(audio_seqlens[ai])))
            ai += 1
            i += 3
        elif input_ids[i] in (AUDIO_PAD, AUDIO_END):
            raise ValueError("stray audio token")
        else:
            i += 1

    # 逐段记账
    segments = []  # list of (n_tokens, start_pos)
    st_idx = 0
    st = 0  # 已消费的输入 token 数
    for pad_index, mel_len in blocks:
        text_len = pad_index - st
        if text_len != 0:
            segments.append((text_len, st_idx))
            st_idx += text_len
        # bos
        segments.append((1, st_idx))
        st_idx += 1
        # audio body
        a_len = audio_expand_len(mel_len)
        segments.append((a_len, st_idx))
        st_idx += a_len
        # eos
        segments.append((1, st_idx))
        st_idx += 1
        st = pad_index + 3
    if st < len(input_ids):
        segments.append((len(input_ids) - st, st_idx))

    row = []
    for n, start in segments:
        row.extend(range(start, start + n))
    # 官方 input_ids 已展开，delta 按展开后长度计
    mrope_delta = (max(row) + 1 - len(row)) if row else 0
    return {"position_ids": [list(row), list(row), list(row)],
            "mrope_delta": mrope_delta}
