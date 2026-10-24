import torch


def _oracle(scores, chunk_length, compression_ratio):
    """独立纯 Python：逐段排序取 top，段顺序拼接。"""
    H, L = scores.shape
    out = []
    for h in range(H):
        flat = []
        start = 0
        while start < L:
            seg_len = min(chunk_length, L - start)
            n_kept = max(1, int(seg_len * (1 - compression_ratio)))
            order = sorted(
                range(seg_len),
                key=lambda j: (-float(scores[h][start + j]), j),
            )
            flat.extend(order[k] + start for k in range(n_kept))
            start += seg_len
        out.append(flat)
    return torch.tensor(out, dtype=torch.long)


def check_example(solve):
    scores = torch.tensor([
        [0.1, 0.9, 0.3, 0.2, 0.8, 0.05, 0.7, 0.6],
        [0.5, 0.4, 0.9, 0.1, 0.05, 0.7, 0.6, 0.8],
    ])
    out = solve(scores, 4, 0.5)
    exp = torch.tensor([[1, 2, 4, 6], [2, 0, 7, 5]])
    assert torch.equal(out, exp), out


def check_matches_oracle(solve):
    torch.manual_seed(1)
    for _ in range(40):
        H = int(torch.randint(1, 6, (1,)))
        L = int(torch.randint(1, 20, (1,)))
        c = int(torch.randint(1, L + 1, (1,)))
        cr = float(torch.rand(1)) * 0.8
        scores = torch.rand(H, L)
        got = solve(scores, c, cr)
        exp = _oracle(scores, c, cr)
        assert torch.equal(got, exp), (H, L, c, cr, got, exp)


def check_last_chunk_short(solve):
    # L=7, chunk=4 → 第二段只有 3 个，n_kept=max(1,int(3*0.5))=1
    scores = torch.tensor([[0.1, 0.9, 0.3, 0.2, 0.8, 0.7, 0.75]])
    out = solve(scores, 4, 0.5)
    exp = torch.tensor([[1, 2, 4]])
    assert torch.equal(out, exp), out


def check_uniform_coverage(solve):
    # 每个 chunk 至少贡献一个保留位（保底 1），高分集中在末段也不会漏掉前段
    scores = torch.tensor([[0.01, 0.02, 0.99, 0.98, 0.97, 0.96]])
    out = solve(scores, 3, 0.5)
    kept = set(out[0].tolist())
    assert any(p < 3 for p in kept), out  # 前段仍保留到位置
    assert any(p >= 3 for p in kept), out  # 后段也保留


check_example.example = True
