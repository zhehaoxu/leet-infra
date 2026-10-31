def solve(seq_len, cp_degree, layout="zigzag"):
    """中文契约 / English contract
    seq_len 是 python int（bool 不算），1 <= seq_len <= 65536；cp_degree 是 python int（bool 不算），
    1 <= cp_degree <= 256；layout 是字符串 "contiguous" 或 "zigzag"（区分大小写）。
    非法入参（类型不符、越界、layout 拼错）一律抛 ValueError / raise ValueError on invalid input.
    不修改输入 / mutate nothing.

    记 C = cp_degree，对齐粒度 align = 2C，补齐长度 padded = ceil(seq_len / align) * align，
    每块 chunk = padded / (2C)。全局补齐序列切成 2C 块，第 c 块覆盖原序列下标
    [c*chunk, (c+1)*chunk)，超出 seq_len 的位置是 padding。
    - layout="zigzag"：rank r 持有第 r 块与第 2C-1-r 块 / rank r owns chunks r and 2C-1-r.
    - layout="contiguous"：rank r 持有第 2r 块与第 2r+1 块 / rank r owns chunks 2r and 2r+1.
    每个 rank 的局部序列 = 前一块在前、后一块在后，长度 2*chunk。
    局部位置里的原序列下标照抄（0 起），padding 记为 -1 /
        local entries are original indices (0-based); padding is -1.

    因果负载 causal_pairs[r]：rank r 每个真实 token（原序列下标 i）在因果注意力下看见
    i+1 个 key（原序列里位置 <= i 的 token），把该 rank 所有真实 token 的 i+1 求和 /
        causal_pairs[r] = sum(i + 1) over every real token i owned by rank r.
    mask 是与 local_indices 同形状的 bool 列表，真实 token 为 True、padding 为 False /
        local_mask[r][i] is True iff local_indices[r][i] >= 0.

    imbalance = max(causal_pairs) / min(causal_pairs)；若 min 为 0，则所有 rank 都为 0 时给 1.0，
    否则给 float("inf") / ratio of max to min causal_pairs, with the zero-min convention above.

    返回四元组 / Returns a 4-tuple:
      (local_indices: list[list[int]] shape C x 2*chunk,
       local_mask: list[list[bool]] shape C x 2*chunk,
       causal_pairs: list[int] length C,
       imbalance: float)
    """
    raise NotImplementedError
