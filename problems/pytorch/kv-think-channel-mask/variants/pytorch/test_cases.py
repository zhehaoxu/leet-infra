import torch


def _oracle(queries, keys, cr):
    """独立：手工算 energy 乘积，用 sorted 索引做 tie-break。"""
    H, W, D = queries.shape
    L = keys.shape[1]
    n_pruned = int(D * cr)
    out = keys.clone().float()
    for h in range(H):
        qn = []
        for d in range(D):
            s = sum(float(queries[h, w, d]) ** 2 for w in range(W)) / W
            qn.append(s)
        kn = []
        for d in range(D):
            s = sum(float(keys[h, l, d]) ** 2 for l in range(L)) / L
            kn.append(s)
        sc = [qn[d] * kn[d] for d in range(D)]
        order = sorted(range(D), key=lambda d: (sc[d], d))[:n_pruned]
        for d in order:
            for l in range(L):
                out[h, l, d] = 0
    return out


def check_example(solve):
    queries = torch.tensor([
        [[1.0, 0.5, 0.0, 0.5], [0.5, 1.0, 0.5, 0.0]],
        [[0.0, 0.0, 1.0, 1.0], [1.0, 0.5, 0.5, 0.0]],
    ])
    keys = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5], [-1.0, 1.0, -1.0, 1.0]],
        [[2.0, 0.0, 0.0, 4.0], [0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 2.0, 3.0]],
    ])
    out = solve(queries, keys, 0.5)
    exp = torch.tensor([
        [[0.0, 2.0, 0.0, 4.0], [0.0, 0.5, 0.0, 0.5], [0.0, 1.0, 0.0, 1.0]],
        [[0.0, 0.0, 0.0, 4.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 2.0, 3.0]],
    ])
    assert torch.allclose(out, exp, atol=1e-5), out


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(30):
        H = int(torch.randint(1, 4, (1,)))
        W = int(torch.randint(1, 6, (1,)))
        L = int(torch.randint(1, 8, (1,)))
        D = int(torch.randint(2, 12, (1,)))
        cr = float(torch.randint(1, D, (1,))) / D
        queries = torch.randn(H, W, D)
        keys = torch.randn(H, L, D)
        got = solve(queries, keys, cr)
        exp = _oracle(queries, keys, cr)
        assert torch.allclose(got, exp, atol=1e-5), (H, W, L, D, cr)


def check_zero_ratio_no_change(solve):
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 5, 4)
    out = solve(q, k, 0.0)
    assert torch.allclose(out, k), out


def check_pruned_channel_count(solve):
    # n_pruned = int(D * cr) 通道被清零
    D = 8
    q = torch.randn(1, 4, D)
    k = torch.randn(1, 6, D)
    for cr, exp_p in [(0.25, 2), (0.5, 4), (0.75, 6)]:
        out = solve(q, k, cr)
        # 找出全零的通道数
        zero_channels = int((out.abs().sum(dim=(0, 1, 2))[0] == 0).sum()) if False else \
            int(((out.abs().sum(dim=(0, 1)) ) == 0).sum())
        assert zero_channels == exp_p, (cr, exp_p, zero_channels)


def check_only_pruned_channels_zeroed(solve):
    q = torch.tensor([[[1.0, 0.5], [2.0, 0.25]]])  # H=1, W=2, D=2
    k = torch.tensor([[[3.0, 1.0], [4.0, 0.5]]])   # H=1, L=2, D=2
    # q_norm = [(1+4)/2, (0.25+0.0625)/2] = [2.5, 0.15625]
    # k_norm = [(9+16)/2, (1+0.25)/2] = [12.5, 0.625]
    # key_scores = [31.25, 0.09766] → 剪通道 1（最小）
    out = solve(q, k, 0.5)
    exp = torch.tensor([[[3.0, 0.0], [4.0, 0.0]]])
    assert torch.allclose(out, exp, atol=1e-5), out


check_example.example = True
