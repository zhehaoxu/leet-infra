"""Qwen2-Audio：音频占位 token 展开数账（solution）。"""

AUDIO_TOKEN = 2


def conv_out_len(mel_len: int) -> int:
    return ((mel_len - 1) // 2 + 1 - 2) // 2 + 1


def solve(input_ids, feature_attention_mask):
    if not isinstance(input_ids, list) or not all(
        isinstance(t, int) and t in (0, AUDIO_TOKEN) for t in input_ids
    ):
        raise ValueError("input_ids must be a list of 0 / AUDIO_TOKEN ints")
    if not isinstance(feature_attention_mask, list) or not feature_attention_mask:
        raise ValueError("feature_attention_mask must be a non-empty list")
    for row in feature_attention_mask:
        if not isinstance(row, list) or not all(
            isinstance(v, int) and v in (0, 1) for v in row
        ):
            raise ValueError("mask rows must be lists of 0/1 ints")

    pooled_lens = [sum(row) for row in feature_attention_mask]
    if any(v < 1 for v in pooled_lens):
        raise ValueError("each audio needs at least 1 valid mel frame")

    n_placeholders = sum(1 for t in input_ids if t == AUDIO_TOKEN)
    if n_placeholders != len(feature_attention_mask):
        raise ValueError(
            f"placeholder count {n_placeholders} != audio count "
            f"{len(feature_attention_mask)}"
        )

    audio_lens = [conv_out_len(v) for v in pooled_lens]
    expanded_len = len(input_ids) - n_placeholders + sum(audio_lens)
    return {"pooled_lens": pooled_lens, "audio_lens": audio_lens,
            "expanded_len": expanded_len}
