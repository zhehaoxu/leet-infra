import torch


def _oracle(scores, cr, alpha):
    """独立：Python 排序 + tie-break by index。"""
    H, L = scores.shape
    n_kept = int(L * (1 - cr))
    n_safe = int(n_kept * alpha)
    if alpha > 0:
        n_safe = max(1, n_safe) if n_kept > 0 else 0
    s = [[float(scores[h, j]) for j in range(L)] for h in range(H)]
    protected = set()
    if n_safe > 0:
        for h in range(H):
            order = sorted(range(L), key=lambda j: (-s[h][j], j))[:n_safe]
            for j in order:
                s[h][j] = float('1e38')  # 有限大、与 FLT_MAX 等价
                protected.add((h, j))
    n_pruned = H * (L - n_kept)
    if n_pruned <= 0:
        return torch.ones(H, L, dtype=torch.bool)
    # 全局排序：按分数升序、索引优先
    flat = [(s[h][j], h, j) for h in range(H) for j in range(L)]
    flat.sort(key=lambda x: (x[0], x[1], x[2]))
    pruned = set((h, j) for _, h, j in flat[:n_pruned])
    mask = torch.zeros(H, L, dtype=torch.bool)
    for h in range(H):
        for j in range(L):
            mask[h, j] = (h, j) not in pruned
    return mask


def check_example_safeguard(solve):
    scores = torch.tensor([[0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
                           [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]])
    out = solve(scores, 0.5, 0.5)
    exp = torch.tensor([[True, True, True, True, True, False],
                        [True, False, False, False, False, False]])
    assert torch.equal(out, exp), out


def check_example_no_safeguard(solve):
    scores = torch.tensor([[0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
                           [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]])
    out = solve(scores, 0.5, 0.0)
    exp = torch.tensor([[True, True, True, True, True, True],
                        [False, False, False, False, False, False]])
    assert torch.equal(out, exp), out


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(30):
        H = int(torch.randint(1, 4, (1,)))
        L = int(torch.randint(2, 10, (1,)))
        cr = float(torch.randint(1, 8, (1,))) / 10.0
        alpha = float(torch.randint(0, 6, (1,))) / 10.0
        scores = torch.rand(H, L)
        got = solve(scores, cr, alpha)
        exp = _oracle(scores, cr, alpha)
        assert torch.equal(got, exp), (H, L, cr, alpha, got, exp)


def check_total_kept_count(solve):
    # 无论 alpha 如何，kept 总数 = H * n_kept
    scores = torch.rand(3, 8)
    for cr in (0.0, 0.25, 0.5, 0.75):
        for alpha in (0.0, 0.1, 0.5):
            mask = solve(scores, cr, alpha)
            n_kept = int(8 * (1 - cr))
            assert int(mask.sum().item()) == 3 * n_kept, (cr, alpha, mask)


def check_no_compression_keeps_all(solve):
    scores = torch.rand(2, 5)
    mask = solve(scores, 0.0, 0.5)
    assert bool(mask.all()), mask


check_example_safeguard.example = True
