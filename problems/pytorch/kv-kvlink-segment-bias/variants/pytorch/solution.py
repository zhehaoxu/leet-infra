import torch


def _as_int(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是 int")
    if value < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}")
    return value


def solve(chunk_lengths, prefix_len=0, reencode_num=5, pad_to=None):
    """KVLink 的段落掩码与 link token 布局（segment-0 目标全局可见）。"""
    if not isinstance(chunk_lengths, (list, tuple)):
        raise ValueError("chunk_lengths 必须是 list/tuple")
    lengths = []
    for i, value in enumerate(chunk_lengths):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"chunk_lengths[{i}] 必须是 int")
        if value < 1:
            raise ValueError(f"chunk_lengths[{i}] 必须 >= 1")
        lengths.append(value)
    prefix_len = _as_int("prefix_len", prefix_len, 0)
    reencode_num = _as_int("reencode_num", reencode_num, 0)

    total = prefix_len + sum(length + reencode_num for length in lengths)
    if total < 1:
        raise ValueError("序列长度必须 >= 1")
    if pad_to is not None:
        pad_to = _as_int("pad_to", pad_to, 0)
        if pad_to < total:
            raise ValueError("pad_to 不能小于真实长度")

    segment_ids = [0] * prefix_len
    for index, length in enumerate(lengths):
        segment_ids.extend([index + 1] * length)
        segment_ids.extend([0] * reencode_num)
    if pad_to is not None:
        segment_ids.extend([-1] * (pad_to - total))

    seg = torch.tensor(segment_ids, dtype=torch.int64)
    length = seg.numel()
    bias = torch.zeros((length, length), dtype=torch.float32)

    causal = torch.ones((length, length), dtype=torch.bool).triu(diagonal=1)
    bias.masked_fill_(causal, float("-inf"))

    target = seg.unsqueeze(1)
    source = seg.unsqueeze(0)
    invalid = (target == -1) | (source == -1)
    global_reader = target == 0
    blocked = invalid | ((target != source) & ~global_reader)
    bias.masked_fill_(blocked, float("-inf"))

    return {"segment_ids": seg, "bias": bias}
