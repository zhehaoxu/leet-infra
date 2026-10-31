"""参考答案：变长打包序列的 zigzag 布局计划。

每个变长段先补到 2C 的整数倍，再把段内切成 2C 个等长块；rank r 拿第 r 块与第 2C-1-r 块，
局部顺序是「前块 + 后块」。padding 位置写 -1。
"""


def _check(cu_seqlens, cp_degree):
    if not isinstance(cu_seqlens, (list, tuple)):
        raise ValueError("cu_seqlens must be a list or tuple of ints")
    if len(cu_seqlens) < 2:
        raise ValueError("cu_seqlens must have at least two entries")
    if isinstance(cp_degree, bool) or not isinstance(cp_degree, int):
        raise ValueError("cp_degree must be a python int")
    if not 1 <= cp_degree <= 256:
        raise ValueError("cp_degree must be in [1, 256]")
    for i, value in enumerate(cu_seqlens):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("cu_seqlens entries must be python ints")
        if value < 0:
            raise ValueError("cu_seqlens entries must be non-negative")
        if i == 0 and value != 0:
            raise ValueError("cu_seqlens must start at 0")
        if i > 0 and value <= cu_seqlens[i - 1]:
            raise ValueError("cu_seqlens must be strictly increasing")
    if cu_seqlens[-1] > 65536:
        raise ValueError("total token count must be <= 65536")


def solve(cu_seqlens, cp_degree):
    _check(cu_seqlens, cp_degree)
    c = cp_degree
    align = 2 * c

    lengths = [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)]
    padded = [((n + align - 1) // align) * align for n in lengths]
    total = sum(padded)
    width = total // c

    cu_padded = [0]
    for p in padded:
        cu_padded.append(cu_padded[-1] + p)

    rank_order = [[-1] * width for _ in range(c)]
    pad_start = 0
    orig_start = 0
    for length, padded_len in zip(lengths, padded):
        chunk = padded_len // (2 * c)
        base = pad_start // c
        for cid in range(2 * c):
            rank = cid if cid < c else 2 * c - 1 - cid
            slot = 0 if cid < c else 1
            for k in range(chunk):
                offset = cid * chunk + k
                rank_order[rank][base + slot * chunk + k] = (
                    orig_start + offset if offset < length else -1
                )
        pad_start += padded_len
        orig_start += length

    pad_total = total - orig_start
    pad_between = any(p != n for n, p in zip(lengths, padded))
    return cu_padded, rank_order, pad_total, bool(pad_between)
