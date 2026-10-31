import torch

KEYS = {"scores", "best", "pattern", "mask"}
TOL = {torch.float32: 1e-5, torch.float64: 1e-11}
LOWER = {}


def _lower(n):
    if n not in LOWER:
        LOWER[n] = torch.tril(torch.ones(n, n, dtype=torch.bool))
    return LOWER[n]


def _mask(kind, a, b, qn, kn, attn, partial_div=False):
    """torch 独立重算三种掩码；partial_div 只改变 block 最后一个不满块的除数。"""
    num_heads, n, dim = qn.shape
    idx = torch.arange(n)
    if kind == "stream_llm":
        band = (idx[None, :] <= idx[:, None]) & (idx[None, :] >= idx[:, None] - b)
        return ((band | (idx[None, :] < a)) & _lower(n)).expand(num_heads, n, n)
    if kind == "vertical_and_slash":
        out = torch.zeros(num_heads, n, n, dtype=torch.bool)
        tail = attn[:, n - 64:, :]
        vertical = tail.sum(1)
        vertical[:, :30] = torch.inf
        diag = torch.zeros(num_heads, n, dtype=attn.dtype)
        for j in range(n):
            delta = j + 1 - n
            lo = max(0, -delta)
            if lo < 64:
                rows = torch.arange(lo, 64)
                diag[:, j] = tail[:, rows, rows + delta].sum(1)
        diag[:, n - 30:] = torch.inf
        for h in range(num_heads):
            drop = sorted(range(n), key=lambda j: (vertical[h, j].item(), j))[: n - a]
            keep = torch.ones(n, dtype=torch.bool)
            keep[drop] = False
            out[h] = keep[None, :] & _lower(n)
            for j in sorted(range(n), key=lambda j: (-diag[h, j].item(), j))[:b]:
                off = j - (n - 1)
                for i in range(max(0, -off), min(n, n - off)):
                    out[h, i, i + off] = True
        return out
    nb = (n - 1) // 32 + 1
    pad = nb * 32
    den = torch.full((nb,), 32.0, dtype=qn.dtype)
    if partial_div:                                      # 对照分支：末块按真实条数归一化
        den[-1] = n - (nb - 1) * 32
    bq = torch.zeros(num_heads, pad, dim, dtype=qn.dtype)
    bk = torch.zeros(num_heads, pad, dim, dtype=kn.dtype)
    bq[:, :n], bk[:, :n] = qn, kn
    bq = bq.reshape(num_heads, nb, 32, dim).sum(2) / den[:, None]
    bk = bk.reshape(num_heads, nb, 32, dim).sum(2) / den[:, None]
    qk = bq @ bk.transpose(1, 2)
    grid = torch.arange(nb)
    qk[:, grid[:, None] < grid[None, :]] = -torch.inf
    out = torch.ones(num_heads, pad, pad, dtype=torch.bool)
    for h in range(num_heads):
        for r in range(nb):
            for c in sorted(range(nb), key=lambda c: (qk[h, r, c].item(), c))[
                    : nb - nb // a]:
                out[h, r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = False
    return (out & torch.tril(torch.ones(pad, pad, dtype=torch.bool)))[:, :n, :n]


def _oracle(q, k, cands, partial_div=False):
    qn, kn = q.to(torch.float64), k.to(torch.float64)
    num_heads, n, dim = qn.shape
    idx = torch.arange(n)
    logits = qn @ kn.transpose(1, 2) / (dim ** 0.5)
    logits = logits.masked_fill(idx[:, None] < idx[None, :], -torch.inf)
    weight = torch.exp(logits - logits.max(-1, keepdim=True).values)
    attn = weight / weight.sum(-1, keepdim=True)
    masks = [_mask(kind, a, b, qn, kn, attn, partial_div) for kind, a, b in cands]
    scores = torch.stack([(attn * m).sum(-1).mean(-1) for m in masks], 1)
    best = scores.argmax(1)
    pattern = torch.tensor([list((1000, 6096) if cands[b][0] == "block_sparse"
                                 else cands[b][1:]) for b in best.tolist()],
                           dtype=torch.int64).reshape(num_heads, 2)
    return scores, best, pattern, torch.stack(
        [masks[b][h] for h, b in enumerate(best.tolist())])


def _verify(solve, q, k, cands, partial_div=False):
    before_q, before_k = q.clone(), k.clone()
    got = solve(q, k, cands)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    assert torch.equal(q, before_q) and torch.equal(k, before_k), "输入被就地修改了"
    scores, best, pattern, mask = _oracle(q, k, cands, partial_div)
    assert got["scores"].dtype == q.dtype and got["scores"].shape == scores.shape
    torch.testing.assert_close(got["scores"], scores.to(q.dtype),
                               rtol=TOL[q.dtype], atol=TOL[q.dtype])
    assert got["best"].dtype == torch.int64 and torch.equal(got["best"], best), (
        f"best 应为 argmax（同分取更小下标），want {best.tolist()}")
    assert got["pattern"].dtype == torch.int64 and torch.equal(
        got["pattern"], pattern), f"pattern 应为 {pattern.tolist()}"
    assert got["mask"].dtype == torch.bool and torch.equal(got["mask"], mask), (
        "mask 必须是胜出候选的因果掩码")
    return got


def check_example(solve):
    """题面示例：q=0 时注意力在因果前缀上均匀，三个候选的掩码可直接手算。"""
    q = torch.zeros(1, 64, 4, dtype=torch.float64)
    cands = [("stream_llm", 30, 8), ("vertical_and_slash", 30, 30), ("block_sparse", 8, 1)]
    got = _verify(solve, q, torch.zeros_like(q), cands)
    assert torch.allclose(got["scores"], torch.tensor(
        [[0.9081807300938489, 0.9975192100819044, 0.0]], dtype=torch.float64))
    assert got["best"].tolist() == [1] and got["pattern"].tolist() == [[30, 30]]
    row = got["mask"][0, 63].nonzero().flatten().tolist()
    assert row == list(range(30)) + list(range(34, 64)), f"最后一行应为 30 列 + 最近 30 对角线：{row}"


def check_vertical_force_keeps_first_30(solve):
    """vertical 前 30 列被置 +inf 强制保留；vertical_size < 30 时它们也会被吃掉一部分。"""
    q = torch.zeros(1, 64, 2, dtype=torch.float64)
    z = torch.zeros_like(q)
    keep = _verify(solve, q, z, [("vertical_and_slash", 30, 1)])["mask"][0]
    assert keep[63].nonzero().flatten().tolist() == list(range(30)) + [34], "前 30 列必须保留"
    cut = _verify(solve, q, z, [("vertical_and_slash", 29, 1)])["mask"][0]
    cols = cut[63].nonzero().flatten().tolist()
    assert cols == list(range(1, 30)) + [34], f"v<30 时 30 列并列 +inf，按小下标先丢第 0 列：{cols}"
    assert int(cut[0].sum()) == 0, "第 0 行只能看到第 0 列，而它被丢掉了 → 整行被掩"


def check_slash_force_keeps_last_30_diagonals(solve):
    """slash 最后 30 条对角线置 +inf；slash_size < 30 时按小下标取到最老的几条。"""
    q = torch.zeros(1, 64, 2, dtype=torch.float64)
    z = torch.zeros_like(q)
    got = _verify(solve, q, z, [("vertical_and_slash", 30, 30)])["mask"][0]
    assert got[63].nonzero().flatten().tolist() == list(range(30)) + list(range(34, 64))
    narrow = _verify(solve, q, z, [("vertical_and_slash", 30, 2)])["mask"][0]
    assert narrow[63].nonzero().flatten().tolist() == list(range(30)) + [34, 35], (
        "s<30 时 30 条对角线并列 +inf，按小下标先取到 delta=-29,-28")


def check_block_sparse_needs_eight_blocks(solve):
    """block_num < topk_ratio 时 block_num//ratio == 0，一个块都不留，分数为 0。"""
    q = torch.randn(1, 64, 3, dtype=torch.float64)
    got = _verify(solve, q, torch.randn(1, 64, 3, dtype=torch.float64), [("block_sparse", 8, 1)])
    assert int(got["mask"].sum()) == 0 and got["scores"].item() == 0.0, "2 个块时应该全丢"
    wide = _verify(solve, q, torch.randn(1, 64, 3, dtype=torch.float64), [("block_sparse", 1, 1)])
    assert int(wide["mask"].sum()) == 64 * 65 // 2, "ratio=1 时不丢块，退化成全因果"


def check_block_padding_enters_denominator(solve):
    """最后一个不满 32 的块补零后仍除以 32（不重归一化），这会改变块的选择。"""
    q = torch.zeros(1, 250, 1, dtype=torch.float64)
    q[0, 224:] = 1.0                                       # 最后一块只有 26 行
    k = torch.cat([torch.full((1, 224, 1), 0.9, dtype=torch.float64),
                   torch.ones(1, 26, 1, dtype=torch.float64)], dim=1)
    got = _verify(solve, q, k, [("block_sparse", 8, 1)])
    real_mean = _oracle(q, k, [("block_sparse", 8, 1)], partial_div=True)[3]
    assert not torch.equal(real_mean, got["mask"]), "按真实条数归一化会选到另一个块"
    assert not got["mask"][0, 240, 240].item(), "补零除以 32 把最后块压成最低分，它被丢掉"
    assert real_mean[0, 240, 240].item(), "按 26 条归一化时最后块分数最高，被丢的是第 0 块"


def check_ties_and_postprocess(solve):
    """同分取更小候选下标；block_sparse 胜出时 (v,s) 被改写成 (1000, 6096)。"""
    q = torch.zeros(1, 64, 2, dtype=torch.float64)
    z = torch.zeros_like(q)
    both = _verify(solve, q, z, [("block_sparse", 1, 1), ("stream_llm", 64, 64)])
    assert both["scores"].tolist() == [[1.0, 1.0]], both["scores"].tolist()
    assert both["best"].tolist() == [0] and both["pattern"].tolist() == [[1000, 6096]]
    flip = _verify(solve, q, z, [("stream_llm", 64, 32), ("block_sparse", 1, 1)])
    assert flip["best"].tolist() == [0] and flip["pattern"].tolist() == [[64, 32]], "stream 保留原尺寸"


def check_dtype_and_mutation(solve):
    """float32 输入 → float32 分数，int64 的 best/pattern；结果与 float64 一致。"""
    torch.manual_seed(4)
    q, k = torch.randn(2, 96, 5), torch.randn(2, 96, 5)
    cands = [("stream_llm", 20, 10), ("vertical_and_slash", 40, 30)]
    lo = _verify(solve, q, k, cands)
    hi = _verify(solve, q.double(), k.double(), cands)
    assert lo["scores"].dtype == torch.float32 and lo["best"].dtype == torch.int64
    torch.testing.assert_close(lo["scores"], hi["scores"].float(), rtol=1e-5, atol=1e-5)
    assert torch.equal(q, q.clone()), "输入被就地修改了"


def check_random_against_oracle(solve):
    """随机形状 × 多种候选网格（含多 head、非连续视图）与 torch oracle 全键对拍。"""
    for seed, (n, heads, dim) in enumerate([(96, 2, 4), (250, 1, 6), (128, 1, 3)]):
        torch.manual_seed(300 + seed)
        q = torch.randn(heads, n + 1, dim, dtype=torch.float64)[:, :n]
        k = torch.randn(heads, n, dim, dtype=torch.float64)
        cands = [("stream_llm", 20, 8), ("vertical_and_slash", min(40, n), 30),
                 ("block_sparse", 8, 1), ("block_sparse", 2, 1)]
        _verify(solve, q, k, cands)
    torch.manual_seed(77)
    q, k = torch.randn(2, 65, 2, dtype=torch.float64), torch.randn(2, 65, 2, dtype=torch.float64)
    _verify(solve, q, k, [("stream_llm", 20, 8), ("vertical_and_slash", 40, 30)])


def check_value_errors(solve):
    q = torch.randn(1, 64, 4)
    good = [("stream_llm", 10, 5)]
    bad = [
        (torch.randn(64, 4), q, good),                       # 非三维
        (q, torch.randn(1, 32, 4), good),                    # N < 64
        (q, q.double(), good),                               # dtype 不一致
        (q, torch.randn(2, 64, 4), good),                    # 形状不一致
        (q, q, []),                                          # 候选为空
        (q, q, [("dense", 1, 1)]),                           # 未知 kind
        (q, q, [("vertical_and_slash", 65, 1)]),             # 尺寸越界
        (q, q, [("block_sparse", 0, 1)]),                    # topk_ratio < 1
    ]
    for i, args in enumerate(bad):
        try:
            solve(*args)
        except ValueError:
            continue
        raise AssertionError(f"第 {i} 个非法输入应抛 ValueError")


check_example.example = True
