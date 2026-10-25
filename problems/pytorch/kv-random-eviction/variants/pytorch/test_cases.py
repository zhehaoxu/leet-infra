import torch

DRAWS = torch.tensor([0.6, 0.1, 0.8, 0.3, 0.9, 0.4, 0.05, 0.7])
ATTN = torch.tensor([0.30, 0.20, 0.10, 0.05, 0.15, 0.05, 0.08, 0.07])


def _oracle(draws, attn_weights, n_sink, window_size, cr):
    """独立向量化 oracle：bool forced 数组 + stable sort（与 solution 的 set/循环路径不同）。"""
    L = draws.shape[0]
    n_kept = max(1, int(L * (1 - cr)))
    forced = torch.zeros(L, dtype=torch.bool)
    forced[:min(n_sink, L)] = True
    if window_size > 0:
        forced[max(L - window_size, 0):] = True
    forced_n = int(forced.sum())
    mid_idx = torch.nonzero(~forced).flatten()
    q = min(max(0, n_kept - forced_n), mid_idx.numel())
    chosen = torch.empty(0, dtype=torch.long)
    if q > 0:
        order = torch.sort(-draws[mid_idx], stable=True).indices[:q]
        chosen = mid_idx[order]
    kept = torch.sort(torch.cat([torch.nonzero(forced).flatten(), chosen])).values
    total = float(attn_weights.sum())
    cov = float(attn_weights[kept].sum()) / total if total > 0 else kept.numel() / L
    return kept, cov


def check_example(solve):
    kept, cov = solve(DRAWS, ATTN, 2, 2, 0.375)
    assert torch.equal(kept, torch.tensor([0, 1, 4, 6, 7])), kept
    assert abs(cov - 0.8) <= 1e-5, cov


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(31)
    for _ in range(30):
        L = int(torch.randint(1, 40, (1,)))
        n_sink = int(torch.randint(0, 6, (1,)))
        win = int(torch.randint(0, 8, (1,)))
        cr = float(torch.rand(1)) * 0.9
        draws = torch.rand(L)
        attn = torch.rand(L)
        gk, gc = solve(draws, attn, n_sink, win, cr)
        ek, ec = _oracle(draws, attn, n_sink, win, cr)
        assert torch.equal(gk, ek), (L, n_sink, win, cr, gk, ek)
        assert abs(gc - ec) <= 1e-5, (gc, ec)


def check_window_covers_all(solve):
    # n_sink + window ≥ L：中段为空 → 全保留，coverage = 1
    kept, cov = solve(torch.rand(6), torch.rand(6), 2, 5, 0.5)
    assert torch.equal(kept, torch.arange(6)), kept
    assert abs(cov - 1.0) <= 1e-5, cov


def check_ties_take_leftmost(solve):
    # 全相等 draws：中段 {2,3,4,5} 中按位置小者优先，q=1 → 选 2
    kept, cov = solve(torch.full((8,), 0.5), ATTN, 2, 2, 0.375)
    assert torch.equal(kept, torch.tensor([0, 1, 2, 6, 7])), kept
    assert abs(cov - 0.75) <= 1e-5, cov


def check_zero_attention_fallback(solve):
    # Σattn = 0 → coverage = |kept|/L = 5/8
    kept, cov = solve(DRAWS, torch.zeros(8), 2, 2, 0.375)
    assert torch.equal(kept, torch.tensor([0, 1, 4, 6, 7])), kept
    assert abs(cov - 0.625) <= 1e-5, cov


def check_quota_exceeds_middle(solve):
    # n_kept=8 → q=4 = 中段长度：中段全取，kept = 全集
    kept, cov = solve(DRAWS, ATTN, 2, 2, 0.0)
    assert torch.equal(kept, torch.arange(8)), kept
    assert abs(cov - 1.0) <= 1e-5, cov


def check_forced_over_budget(solve):
    # |F| > n_kept：保护优先，保留数超预算，且不做任何抽签
    kept, cov = solve(DRAWS, ATTN, 3, 3, 0.75)
    assert torch.equal(kept, torch.tensor([0, 1, 2, 5, 6, 7])), kept
