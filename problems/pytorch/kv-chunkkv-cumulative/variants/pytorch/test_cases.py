import torch


def _oracle(scores, chunk_length, compression_ratio):
    """独立纯 Python：逐块累加求和→块均分→按块分挑选→升序展开 token 位置。"""
    H, L = scores.shape
    c = int(chunk_length)
    n_chunks = (L + c - 1) // c
    cs = []
    for i in range(n_chunks):
        lo = i * c
        hi = min(lo + c, L)
        acc = 0.0
        for t in range(lo, hi):
            s = 0.0
            for h in range(H):
                s += float(scores[h][t])
            acc += s
        cs.append(acc / (hi - lo))
    n_kept = max(1, int(n_chunks * (1 - compression_ratio)))
    order = sorted(range(n_chunks), key=lambda i: (-cs[i], i))
    idxs = []
    for i in order[:n_kept]:
        lo = i * c
        hi = min(lo + c, L)
        idxs.extend(range(lo, hi))
    return torch.tensor(sorted(idxs), dtype=torch.long)


def check_example(solve):
    scores = torch.tensor([
        [0.5, 0.4, 0.6, 0.5, 0.05, 0.05, 0.05, 0.05, 0.4, 0.3, 0.5, 0.4],
        [0.4, 0.5, 0.5, 0.4, 0.05, 0.05, 0.05, 0.05, 0.3, 0.4, 0.4, 0.3],
    ])
    out = solve(scores, 4, 1 / 3)
    exp = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11])
    assert torch.equal(out, exp), out


def check_matches_oracle(solve):
    torch.manual_seed(2)
    for _ in range(40):
        H = int(torch.randint(1, 6, (1,)))
        L = int(torch.randint(1, 24, (1,)))
        c = int(torch.randint(1, L + 2, (1,)))
        cr = float(torch.rand(1)) * 0.85
        scores = torch.rand(H, L)
        got = solve(scores, c, cr)
        exp = _oracle(scores, c, cr)
        assert torch.equal(got, exp), (H, L, c, cr, got, exp)


def check_whole_chunks_only(solve):
    # 保留的一定是"整块"：块内 token 要么全选要么全不选
    scores = torch.tensor([
        [0.9, 0.9, 0.1, 0.1, 0.8, 0.8],
        [0.7, 0.7, 0.2, 0.2, 0.6, 0.6],
    ])
    out = solve(scores, 3, 1 / 3)  # N=2, keep 1 → 选块0(sum 更? 块0:1.6,1.6,0.3→mean~1.17; 块1:0.3,1.4,1.4)
    kept = out.tolist()
    assert kept == sorted(kept), kept
    # 只保留一个完整块（3 个连续位置）
    assert len(kept) == 3, kept
    assert kept in ([0, 1, 2], [3, 4, 5]), kept


def check_single_chunk_keep_all(solve):
    # L<=c → 只有一个块，保底全保留
    scores = torch.tensor([[0.1, 0.5, 0.2], [0.3, 0.4, 0.1]])
    out = solve(scores, 5, 0.9)
    assert torch.equal(out, torch.tensor([0, 1, 2])), out


check_example.example = True
