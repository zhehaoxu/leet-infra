import random

import torch


def _oracle(attn, kernel_size):
    """独立：手工 avg_pool——越界补 0、分母恒为 kernel_size。"""
    H, W, P = attn.shape
    base = [[sum(float(attn[h, w, i]) for w in range(W)) / W for i in range(P)] for h in range(H)]
    half = kernel_size // 2
    pooled = []
    for h in range(H):
        row = []
        for i in range(P):
            s = 0.0
            for j in range(i - half, i + half + 1):
                if 0 <= j < P:
                    s += base[h][j]
            row.append(s / kernel_size)
        pooled.append(row)
    m = max(max(r) for r in pooled) + 1.0
    out = torch.zeros(H, P + W, dtype=torch.float32)
    for h in range(H):
        for i in range(P):
            out[h, i] = pooled[h][i]
        for j in range(W):
            out[h, P + j] = m
    return out


def check_example(solve):
    attn = torch.tensor([
        [[0.1, 0.2, 0.05, 0.15, 0.1],
         [0.3, 0.1, 0.1, 0.2, 0.05],
         [0.05, 0.4, 0.1, 0.1, 0.2]],
        [[0.2, 0.1, 0.3, 0.1, 0.05],
         [0.1, 0.2, 0.1, 0.3, 0.2],
         [0.3, 0.1, 0.1, 0.1, 0.25]],
    ])
    out = solve(attn, 3)
    exp = torch.tensor([
        [0.127778, 0.155556, 0.155556, 0.116667, 0.088889, 1.166667, 1.166667, 1.166667],
        [0.111111, 0.166667, 0.155556, 0.166667, 0.111111, 1.166667, 1.166667, 1.166667],
    ])
    assert torch.allclose(out, exp, atol=1e-5), out


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(30):
        H = int(torch.randint(1, 4, (1,)))
        W = int(torch.randint(1, 6, (1,)))
        P = int(torch.randint(2, 12, (1,)))
        K = random.choice([1, 3, 5, 7])
        attn = torch.rand(H, W, P)
        got = solve(attn, K)
        exp = _oracle(attn, K)
        assert torch.allclose(got, exp, atol=1e-5), (H, W, P, K, got, exp)


def check_tail_is_max_plus_one(solve):
    # 无论 attn 怎么变，尾部 W 位都应等于 max(pooled)+1
    attn = torch.rand(2, 3, 5)
    K = 3
    out = solve(attn, K)
    pooled = out[:, :-3]
    tail = out[:, -3:]
    expected = float(pooled.max()) + 1.0
    assert torch.allclose(tail, torch.full_like(tail, expected)), (tail, expected)


def check_identity_kernel(solve):
    # kernel_size=1 时 avg_pool 是恒等；pooled = base = attn.mean(-2)
    attn = torch.tensor([[[0.2, 0.4, 0.6], [0.1, 0.5, 0.3]]])  # (1, 2, 3)
    out = solve(attn, 1)
    base = attn.mean(dim=-2)  # (1, 3) = [0.15, 0.45, 0.45]
    m = float(base.max()) + 1.0
    exp = torch.cat([base, torch.full((1, 2), m)], dim=-1)
    assert torch.allclose(out, exp, atol=1e-6), (out, exp)


check_example.example = True
