def _oracle(n, c, layout):
    """纯 Python：逐个原序列 token 反推 (rank, slot, 局部位置)，再独立累计因果负载。"""
    align = 2 * c
    padded = ((n + align - 1) // align) * align
    chunk = padded // (2 * c)
    width = 2 * chunk

    rows = [[-1] * width for _ in range(c)]
    owner = [0] * n
    for i in range(n):
        cid = i // chunk
        pos = i % chunk
        if layout == "contiguous":
            rank, slot = cid // 2, cid % 2
        else:
            rank = min(cid, 2 * c - 1 - cid)
            slot = 0 if cid < c else 1
        rows[rank][slot * chunk + pos] = i
        owner[i] = rank

    masks = [[v >= 0 for v in row] for row in rows]
    pairs = [0] * c
    for i in range(n):
        pairs[owner[i]] += i + 1

    hi, lo = max(pairs), min(pairs)
    imbalance = (1.0 if hi == 0 else float("inf")) if lo == 0 else hi / lo
    return rows, masks, pairs, float(imbalance)


def _verify(solve, args):
    out = solve(*args)
    assert isinstance(out, tuple) and len(out) == 4
    rows, masks, pairs, imbalance = out
    n, c = args[0], args[1]
    align = 2 * c
    padded = ((n + align - 1) // align) * align
    width = padded // c
    assert len(rows) == c and all(len(r) == width for r in rows)
    assert len(masks) == c and all(len(m) == width for m in masks)
    assert len(pairs) == c
    for row, mask in zip(rows, masks):
        for v, m in zip(row, mask):
            assert m is (v >= 0)
    assert isinstance(imbalance, float)
    return rows, masks, pairs, imbalance


def check_example(solve):
    rows, masks, pairs, imbalance = _verify(solve, (8, 2, "zigzag"))
    assert rows == [[0, 1, 6, 7], [2, 3, 4, 5]], rows
    assert masks == [[True] * 4, [True] * 4], masks
    assert pairs == [18, 18], pairs
    assert imbalance == 1.0, imbalance

    rows, masks, pairs, imbalance = _verify(solve, (8, 2, "contiguous"))
    assert rows == [[0, 1, 2, 3], [4, 5, 6, 7]], rows
    assert pairs == [10, 26], pairs
    assert abs(imbalance - 2.6) < 1e-12, imbalance


check_example.example = True


def check_matches_oracle(solve):
    configs = []
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 31, 32]:
        for c in [1, 2, 3, 4, 8]:
            if c > 16:
                continue
            for layout in ("contiguous", "zigzag"):
                configs.append((n, c, layout))
    for args in configs:
        got = _verify(solve, args)
        want = _oracle(*args)
        assert got[0] == want[0], (args, got[0], want[0])
        assert got[1] == want[1], (args, got[1], want[1])
        assert got[2] == want[2], (args, got[2], want[2])
        assert (got[3] == want[3]) or (got[3] != got[3] and want[3] != want[3]), (
            args, got[3], want[3])


def check_padding_ledger(solve):
    # N=6, C=2: padded=8，尾部 2 个 padding 全部落到 zigzag 的 back 块
    rows, masks, pairs, imbalance = _verify(solve, (6, 2, "zigzag"))
    assert rows == [[0, 1, -1, -1], [2, 3, 4, 5]], rows
    assert masks == [[True, True, False, False], [True] * 4], masks
    assert pairs == [3, 18], pairs
    assert abs(imbalance - 6.0) < 1e-12, imbalance

    # C=1：padded = ceil(N/2)*2，padding 只会出现在最后
    rows, masks, pairs, _ = _verify(solve, (5, 1, "zigzag"))
    assert rows == [[0, 1, 2, 3, 4, -1]], rows
    assert pairs == [15], pairs

    # 每 rank 的 padding 数可以不同，但全库 padding 总数恒为 padded - N
    for n in (1, 3, 5, 6, 7, 9, 14):
        for c in (1, 2, 3, 4):
            for layout in ("contiguous", "zigzag"):
                rows, masks, _, _ = _verify(solve, (n, c, layout))
                pads = sum(1 for row in masks for m in row if not m)
                align = 2 * c
                padded = ((n + align - 1) // align) * align
                assert pads == padded - n, (n, c, layout, pads)
                valid = sorted(v for row in rows for v in row if v >= 0)
                assert valid == list(range(n)), (n, c, layout, valid)


def check_zigzag_balances_causal_work(solve):
    # padded == N 时，zigzag 的前/后块负载严格相等（每提一个 rank，前块 +S*S、后块 -S*S）
    for n, c in [(4, 2), (8, 2), (12, 3), (16, 4), (32, 8), (64, 16), (96, 12)]:
        rows, _, pairs, imbalance = _verify(solve, (n, c, "zigzag"))
        assert imbalance == 1.0, (n, c, pairs)
        assert len(set(pairs)) == 1, (n, c, pairs)

    # contiguous 至少在一半以上的配置里明显失衡；C=1 时两种布局完全一致
    for n, c in [(8, 2), (16, 4), (32, 8)]:
        _, _, pairs, imbalance = _verify(solve, (n, c, "contiguous"))
        assert max(pairs) > min(pairs) and imbalance > 1.0, (n, c, pairs)
    for n in (4, 7, 12):
        a = _verify(solve, (n, 1, "zigzag"))
        b = _verify(solve, (n, 1, "contiguous"))
        assert a[0] == b[0] and a[2] == b[2] and a[3] == b[3]


def check_chunk_composition(solve):
    n, c = 12, 3
    chunk = 2
    for layout in ("contiguous", "zigzag"):
        rows, _, _, _ = _verify(solve, (n, c, layout))
        for rank in range(c):
            if layout == "contiguous":
                front, back = 2 * rank, 2 * rank + 1
            else:
                front, back = rank, 2 * c - 1 - rank
            assert rows[rank][:chunk] == [front * chunk + k for k in range(chunk)]
            assert rows[rank][chunk:] == [back * chunk + k for k in range(chunk)]


def check_empty_ranks(solve):
    # C > N：拿不到真实 token 的 rank 因果负载为 0，且 imbalance 为 inf（只要有 rank 有负载）
    rows, masks, pairs, imbalance = _verify(solve, (2, 4, "zigzag"))
    assert rows == [[0, -1], [1, -1], [-1, -1], [-1, -1]], rows
    assert pairs == [1, 2, 0, 0], pairs
    assert imbalance == float("inf"), imbalance

    # 极端退化：只有 1 个 token 也会被某个 rank 拿到，其余为 0
    _, _, pairs, imbalance = _verify(solve, (1, 8, "contiguous"))
    assert sorted(pairs)[:7] == [0] * 7 and max(pairs) == 1
    assert imbalance == float("inf")


def check_invalid_inputs(solve):
    def expect_error(*args):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError(f"expected ValueError for {args!r}")

    for bad in (0, -1, True, 2.5, "8", None):
        expect_error(bad, 2, "zigzag")
    for bad in (0, -1, True, 2.0, "2", 257):
        expect_error(8, bad, "zigzag")
    for bad in ("ZigZag", "", "ZIGZAG", "contig", None, True):
        expect_error(8, 2, bad)
    expect_error(65537, 1, "zigzag")
