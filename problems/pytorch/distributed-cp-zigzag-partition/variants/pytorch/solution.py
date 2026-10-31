"""参考答案：把序列按 2C 对齐切块，再按 contiguous / zigzag 分给 C 个 rank。

zigzag：rank r 拿第 r 块与第 2C-1-r 块（前后拼接）；contiguous：rank r 拿第 2r 与 2r+1 块。
超出原序列的位置记 -1，因果负载按「原序列下标 i 的查询看见 i+1 个 key」累计。
"""


def _check(seq_len, cp_degree, layout):
    if isinstance(seq_len, bool) or not isinstance(seq_len, int):
        raise ValueError("seq_len must be a python int")
    if not 1 <= seq_len <= 65536:
        raise ValueError("seq_len must be in [1, 65536]")
    if isinstance(cp_degree, bool) or not isinstance(cp_degree, int):
        raise ValueError("cp_degree must be a python int")
    if not 1 <= cp_degree <= 256:
        raise ValueError("cp_degree must be in [1, 256]")
    if not isinstance(layout, str) or layout not in ("contiguous", "zigzag"):
        raise ValueError("layout must be 'contiguous' or 'zigzag'")


def solve(seq_len, cp_degree, layout="zigzag"):
    _check(seq_len, cp_degree, layout)
    n, c = seq_len, cp_degree

    align = 2 * c
    padded = ((n + align - 1) // align) * align
    chunk = padded // (2 * c)

    chunks = []
    for cid in range(2 * c):
        start = cid * chunk
        chunks.append([start + k if start + k < n else -1 for k in range(chunk)])

    local_indices = []
    for rank in range(c):
        if layout == "contiguous":
            first, second = 2 * rank, 2 * rank + 1
        else:
            first, second = rank, 2 * c - 1 - rank
        local_indices.append(chunks[first] + chunks[second])

    local_mask = [[idx >= 0 for idx in row] for row in local_indices]
    causal_pairs = [sum(idx + 1 for idx in row if idx >= 0) for row in local_indices]

    hi, lo = max(causal_pairs), min(causal_pairs)
    if lo == 0:
        imbalance = 1.0 if hi == 0 else float("inf")
    else:
        imbalance = hi / lo
    return local_indices, local_mask, causal_pairs, float(imbalance)
