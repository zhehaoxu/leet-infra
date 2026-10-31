def _oracle(cu, c):
    """纯 Python：每段先拼出补齐流，再用切片按「前块 + 后块」放到对应 rank 的槽里。"""
    lengths = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
    align = 2 * c
    padded = [((n + align - 1) // align) * align for n in lengths]
    width = sum(padded) // c

    rows = [[-1] * width for _ in range(c)]
    cu_padded = [0]
    for p in padded:
        cu_padded.append(cu_padded[-1] + p)

    pad_start = 0
    orig_start = 0
    for length, padded_len in zip(lengths, padded):
        chunk = padded_len // (2 * c)
        stream = [orig_start + o if o < length else -1 for o in range(padded_len)]
        chunks = [stream[i * chunk:(i + 1) * chunk] for i in range(2 * c)]
        base = pad_start // c
        for rank in range(c):
            front = chunks[rank]
            back = chunks[2 * c - 1 - rank]
            rows[rank][base:base + chunk] = front
            rows[rank][base + chunk:base + 2 * chunk] = back
        pad_start += padded_len
        orig_start += length

    pad_total = sum(padded) - sum(lengths)
    return cu_padded, rows, pad_total, any(p != n for n, p in zip(lengths, padded))


def _verify(solve, cu, c):
    out = solve(cu, c)
    assert isinstance(out, tuple) and len(out) == 4
    cu_p, rows, pad_total, flag = out
    n = cu[-1]
    assert isinstance(cu_p, list) and len(cu_p) == len(cu)
    assert cu_p[0] == 0 and cu_p[-1] == sum(
        ((cu[i + 1] - cu[i] + 2 * c - 1) // (2 * c)) * (2 * c) for i in range(len(cu) - 1)
    )
    width = cu_p[-1] // c
    assert len(rows) == c and all(len(r) == width for r in rows)
    assert isinstance(pad_total, int) and pad_total == cu_p[-1] - n
    assert isinstance(flag, bool)
    valid = sorted(v for r in rows for v in r if v >= 0)
    assert valid == list(range(n))
    pads = sum(1 for r in rows for v in r if v < 0)
    assert pads == pad_total
    return out


def check_example(solve):
    cu_p, rows, pad_total, flag = _verify(solve, [0, 3, 5], 2)
    assert cu_p == [0, 4, 8], cu_p
    assert rows == [[0, -1, 3, -1], [1, 2, 4, -1]], rows
    assert pad_total == 3 and flag is True

    cu_p, rows, pad_total, flag = _verify(solve, [0, 4], 2)
    assert cu_p == [0, 4], cu_p
    assert rows == [[0, 3], [1, 2]], rows
    assert pad_total == 0 and flag is False


check_example.example = True


def check_matches_oracle(solve):
    cases = [
        ([0, 1], 1), ([0, 1], 4), ([0, 2], 2), ([0, 3], 2), ([0, 4], 2),
        ([0, 5], 1), ([0, 5], 2), ([0, 5], 3), ([0, 8], 4),
        ([0, 3, 5], 2), ([0, 1, 2, 3], 2), ([0, 2, 6], 1), ([0, 2, 6], 2),
        ([0, 2, 6], 3), ([0, 4, 4 + 1], 2), ([0, 7, 9, 16], 4),
        ([0, 9, 10, 11, 12], 3), ([0, 16], 8), ([0, 1, 17], 4),
    ]
    for cu, c in cases:
        got = _verify(solve, cu, c)
        want = _oracle(cu, c)
        assert got[0] == want[0], (cu, c, got[0], want[0])
        assert got[1] == want[1], (cu, c, got[1], want[1])
        assert got[2] == want[2], (cu, c, got[2], want[2])
        assert got[3] == want[3], (cu, c, got[3], want[3])


def check_single_sequence_matches_zigzag_layout(solve):
    # 单段时，本题的 rank_order 就是 zigzag token 切分的局部下标
    expected = {
        (8, 2): [[0, 1, 6, 7], [2, 3, 4, 5]],
        (6, 2): [[0, 1, -1, -1], [2, 3, 4, 5]],
        (12, 3): [[0, 1, 10, 11], [2, 3, 8, 9], [4, 5, 6, 7]],
        (5, 1): [[0, 1, 2, 3, 4, -1]],
    }
    for (n, c), want in expected.items():
        _, rows, _, _ = _verify(solve, [0, n], c)
        assert rows == want, ((n, c), rows, want)


def check_padding_ledger(solve):
    # 每段补齐到 2C 的倍数，padding 只出现在各段尾部
    for cu, c in [([0, 3, 5], 2), ([0, 7, 9], 4), ([0, 2, 3, 4], 3), ([0, 1], 8)]:
        cu_p, rows, pad_total, flag = _verify(solve, cu, c)
        for b in range(len(cu) - 1):
            length = cu[b + 1] - cu[b]
            padded_len = cu_p[b + 1] - cu_p[b]
            assert padded_len % (2 * c) == 0, (cu, c, b)
            assert padded_len - length < 2 * c, (cu, c, b)
        assert pad_total == cu_p[-1] - cu[-1]
        assert flag is (pad_total > 0), (cu, c, flag, pad_total)

    # 全整除时既不补齐也没有段间 padding
    cu_p, rows, pad_total, flag = _verify(solve, [0, 4, 8, 12], 2)
    assert cu_p == [0, 4, 8, 12] and pad_total == 0 and flag is False
    assert rows == [[0, 3, 4, 7, 8, 11], [1, 2, 5, 6, 9, 10]], rows


def check_rank_assignment(solve):
    # 每个 rank 的前块与后块分别来自第 r、2C-1-r 块
    cu, c = [0, 12, 24], 3
    _, rows, _, _ = _verify(solve, cu, c)
    chunk = 12 // (2 * c)
    assert chunk == 2
    for b, orig in enumerate((0, 12)):
        base = (b * 12) // c
        for rank in range(c):
            front = rows[rank][base]
            back = rows[rank][base + chunk]
            assert front == orig + rank * chunk, (b, rank, front)
            assert back == orig + (2 * c - 1 - rank) * chunk, (b, rank, back)


def check_invalid_inputs(solve):
    def expect_error(cu, c):
        try:
            solve(cu, c)
        except ValueError:
            return
        raise AssertionError(f"expected ValueError for {cu!r}, {c!r}")

    for bad in (None, 0, "0,4", {0: 4}):
        expect_error(bad, 2)
    for bad in ([], [0], [1, 4], [0, 0], [0, 3, 2], [0, -1, 4], [0, 2.0, 4],
                [0, True, 4], [0, 4, 65537], [0, 65536, 65537]):
        expect_error(bad, 2)
    for bad in (0, -1, True, 2.0, "2", 257):
        expect_error([0, 4], bad)
