import torch

ATTN = torch.tensor([
    [[0.40, 0.20, 0.40], [0.10, 0.50, 0.40]],
    [[0.30, 0.30, 0.40], [0.60, 0.20, 0.20]],
    [[0.50, 0.10, 0.40], [0.20, 0.40, 0.40]],
    [[0.25, 0.25, 0.50], [0.55, 0.25, 0.20]],
])


def _oracle(attn, num_kv_heads, cr):
    """独立纯 Python 多重循环 oracle：逐 (kv head, group, 行, 列) 累加（与 solution 的向量化路径不同）。"""
    Hq, W, C = attn.shape
    H = num_kv_heads
    G = Hq // H
    ctx = []
    for h in range(H):
        row = []
        for j in range(C):
            acc = 0.0
            for g in range(G):
                for i in range(W):
                    acc += float(attn[h * G + g, i, j]) * (C + i)
            row.append(acc / (G * W))
        ctx.append(row)
    gmax = max(max(r) for r in ctx)
    pad = gmax + 1.0
    scores = [r + [pad] * W for r in ctx]
    L = C + W
    n_kept = max(1, int(L * (1 - cr)))
    kept = []
    for r in scores:
        order = sorted(range(L), key=lambda j: (-r[j], j))[:n_kept]
        kept.append(sorted(order))
    return torch.tensor(scores, dtype=torch.float32), torch.tensor(kept, dtype=torch.long)


def check_example(solve):
    scores, kept = solve(ATTN, 2, 0.4)
    exp_scores = torch.tensor([
        [1.225, 1.075, 1.2, 2.3125, 2.3125],
        [1.3125, 0.9125, 1.275, 2.3125, 2.3125],
    ])
    exp_kept = torch.tensor([[0, 3, 4], [0, 3, 4]])
    assert torch.allclose(scores, exp_scores, atol=1e-5, rtol=1e-5), scores
    assert torch.equal(kept, exp_kept), kept


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(3)
    for _ in range(30):
        H = int(torch.randint(1, 4, (1,)))
        G = int(torch.randint(1, 4, (1,)))
        W = int(torch.randint(1, 6, (1,)))
        C = int(torch.randint(1, 10, (1,)))
        cr = float(torch.rand(1)) * 0.9
        attn = torch.rand(H * G, W, C)
        gs, gk = solve(attn, H, cr)
        es, ek = _oracle(attn, H, cr)
        assert torch.allclose(gs, es, atol=1e-5, rtol=1e-5), (gs, es)
        assert torch.equal(gk, ek), (cr, gk, ek)


def check_single_window_row(solve):
    # W=1：乘数为常数 C，缩放不改变上下文排序；pad = max+1
    attn = torch.tensor([[[0.5, 0.2, 0.3]], [[0.1, 0.7, 0.2]]])
    scores, kept = solve(attn, 1, 1 / 3)
    exp_scores = torch.tensor([[0.9, 1.35, 0.75, 2.35]])
    assert torch.allclose(scores, exp_scores, atol=1e-5, rtol=1e-5), scores
    assert torch.equal(kept, torch.tensor([[1, 3]])), kept


def check_zero_attention(solve):
    # 全零注意力：上下文分 0，pad=1；n_kept=2 时平局取窗口最左两格
    scores, kept = solve(torch.zeros(1, 3, 2), 1, 0.6)
    assert torch.allclose(scores, torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]]), atol=1e-5, rtol=1e-5), scores
    assert torch.equal(kept, torch.tensor([[2, 3]])), kept


def check_ratio_zero_keeps_all(solve):
    scores, kept = solve(ATTN, 2, 0.0)
    assert scores.shape == (2, 5), scores.shape
    assert torch.equal(kept, torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])), kept


def check_window_alive_invariant(solve):
    # pad = 全局max+1 严格大于任何上下文分 → n_kept ≥ W 时窗口位必然全部存活
    torch.manual_seed(9)
    W = 3
    attn = torch.rand(2, W, 7)
    _, kept = solve(attn, 1, 0.5)   # L=10, n_kept=5 ≥ W
    assert set(range(7, 10)).issubset(set(kept[0].tolist())), kept
