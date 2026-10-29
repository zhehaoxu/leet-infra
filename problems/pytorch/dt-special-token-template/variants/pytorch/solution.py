"""Position-based BERT / RoBERTa template assembly; standard library only."""


def solve(ids_a, ids_b, style, cls_id, sep_id, add_special_tokens=True):
    if style not in ("bert", "roberta"):
        raise ValueError("style must be bert or roberta")

    # Each segment carries explicit provenance, independent of numeric IDs.
    segments = []
    if add_special_tokens:
        segments.append(([cls_id], 0, 1, None))
    segments.append((ids_a, 0, 0, 0))
    if add_special_tokens:
        segments.append(([sep_id], 0, 1, None))
    if ids_b is not None:
        b_type = 1 if style == "bert" else 0
        if add_special_tokens and style == "roberta":
            segments.append(([sep_id], 0, 1, None))
        segments.append((ids_b, b_type, 0, 1))
        if add_special_tokens:
            segments.append(([sep_id], b_type, 1, None))

    result = dict(input_ids=[], token_type_ids=[], special_tokens_mask=[], sequence_ids=[])
    for ids, type_id, special, sequence in segments:
        result["input_ids"].extend(ids)
        result["token_type_ids"].extend([type_id] * len(ids))
        result["special_tokens_mask"].extend([special] * len(ids))
        result["sequence_ids"].extend([sequence] * len(ids))
    return result
