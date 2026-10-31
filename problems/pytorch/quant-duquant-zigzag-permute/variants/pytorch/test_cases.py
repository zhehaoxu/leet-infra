import torch

_EXAMPLE_W = [
    [0.5, 3.0, 1.0, 2.0, 0.25, 4.0],
    [0.1, -2.5, 0.5, -1.0, 0.1, -3.5],
]


def _oracle(W, block_size):
    hidden = len(W[0])
    groups = hidden // block_size
    keys = [max(abs(float(row[c])) for row in W) for c in range(hidden)]

    pairs = [(c, keys[c]) for c in range(hidden)]
    pairs.sort(key=lambda item: item[1], reverse=True)

    buckets = [[] for _ in range(groups)]
    cur, up = 0, True
    for pair in pairs:
        buckets[cur].append(pair)
        if up:
            cur += 1
            if cur == groups:
                cur -= 1
                up = False
        else:
            cur -= 1
            if cur == -1:
                cur += 1
                up = True

    perm = []
    for bucket in buckets:
        bucket.sort(key=lambda item: item[1], reverse=True)
        perm.extend(idx for idx, _ in bucket)
    permuted = [[float(row[c]) for c in perm] for row in W]
    return perm, permuted, keys


def check_example(solve):
    W = torch.tensor(_EXAMPLE_W, dtype=torch.float32)
    out = solve(W, 2)

    assert out["perm"].tolist() == [5, 4, 1, 0, 3, 2], f"got {out['perm'].tolist()}"
    expect = torch.tensor([
        [4.0, 0.25, 3.0, 0.5, 2.0, 1.0],
        [-3.5, 0.1, -2.5, 0.1, -1.0, 0.5],
    ])
    assert torch.equal(out["permuted"], expect), f"got {out['permuted'].tolist()}"


check_example.example = True


def check_permutation_and_copy(solve):
    """perm 必须是 0..H-1 的排列，且 permuted 正好是 W 按 perm 取列。"""
    torch.manual_seed(7)
    W = torch.randn(4, 8)
    out = solve(W, 4)
    perm = out["perm"].tolist()
    assert sorted(perm) == list(range(8)), f"不是排列: {perm}"
    assert torch.equal(out["permuted"], W[:, torch.tensor(perm)]), "重排结果不符"


def check_zigzag_pattern(solve):
    """排序后的列必须按蛇形轮流进入各块：前 G 名分属 G 个不同块，
    接下来的 G 名以相反块序再各进一块。"""
    W = torch.tensor(_EXAMPLE_W, dtype=torch.float32)
    out = solve(W, 2)
    perm = out["perm"].tolist()
    _, _, keys = _oracle(_EXAMPLE_W, 2)
    order = sorted(range(6), key=lambda c: (-keys[c], c))

    def bucket_of(col):
        return perm.index(col) // 2

    assert [bucket_of(order[i]) for i in range(3)] == [0, 1, 2], "首轮不是 0,1,2"
    assert [bucket_of(order[i]) for i in range(3, 6)] == [2, 1, 0], "次轮不是 2,1,0"


def check_matches_oracle(solve):
    """多种块大小/形状与纯 Python oracle 对齐。"""
    torch.manual_seed(13)
    cases = [
        (torch.randn(5, 8) * 2, 4),
        (torch.randn(3, 12), 3),
        (torch.randn(4, 6), 1),
        (torch.randn(2, 4), 4),
        (torch.rand(3, 10) * 4 - 2, 5),
    ]
    for idx, (W, bsz) in enumerate(cases):
        perm, permuted, _ = _oracle(W.tolist(), bsz)
        out = solve(W, bsz)
        assert out["perm"].tolist() == perm, f"case{idx} perm: {out['perm'].tolist()} != {perm}"
        assert torch.allclose(out["permuted"],
                              torch.tensor(permuted, dtype=torch.float32), atol=1e-6), \
            f"case{idx} permuted 不符"


def check_ties_stable(solve):
    """键值相等时保持原下标升序（稳定排序）。"""
    W = torch.tensor([
        [2.0, 1.0, 2.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
    ])
    out = solve(W, 2)
    assert out["perm"].tolist() == [0, 3, 2, 1], f"got {out['perm'].tolist()}"

    # B=1 时退化为全局稳定排序；B=H 时整块内部稳定排序
    assert solve(W, 1)["perm"].tolist() == [0, 2, 1, 3]
    assert solve(W, 4)["perm"].tolist() == [0, 2, 1, 3]


def check_block_order(solve):
    """每个块内部必须按列键值降序。"""
    torch.manual_seed(29)
    W = torch.randn(6, 12) * 3
    out = solve(W, 3)
    perm = out["perm"].tolist()
    keys = [max(abs(float(v)) for v in W[:, c]) for c in range(12)]
    for start in range(0, 12, 3):
        block = perm[start:start + 3]
        vals = [keys[c] for c in block]
        assert all(vals[i] >= vals[i + 1] for i in range(2)), \
            f"块 {start // 3} 内部未降序: {vals}"


def check_validation(solve):
    """形状、类型、block_size 非法时抛 ValueError。"""
    good = torch.zeros(2, 6)

    def expect_error(fn, tag):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError：{tag}")

    expect_error(lambda: solve(torch.zeros(6), 2), "W 一维")
    expect_error(lambda: solve(torch.zeros(2, 3, 2), 2), "W 三维")
    expect_error(lambda: solve(torch.zeros(0, 6), 2), "空行")
    expect_error(lambda: solve(torch.zeros(2, 0), 2), "空列")
    expect_error(lambda: solve(torch.zeros(2, 6, dtype=torch.int64), 2), "W 非浮点")
    expect_error(lambda: solve(torch.full((2, 6), float("nan")), 2), "W 含 nan")
    expect_error(lambda: solve(good, 0), "block_size 为 0")
    expect_error(lambda: solve(good, -1), "block_size 为负")
    expect_error(lambda: solve(good, 7), "block_size 超过列数")
    expect_error(lambda: solve(good, 4), "列数不能被整除")
    expect_error(lambda: solve(good, 2.0), "block_size 非整数")
    expect_error(lambda: solve(good, True), "block_size 为 bool")
