import torch


def _oracle(attn, num_kv_heads):
    """独立：Python 逐元素算 mean → 复制到 H 行 → 追加 max+1。"""
    Hq, L, _ = attn.shape
    row = []
    for j in range(L - 1):
        s = 0.0
        for h in range(Hq):
            s += float(attn[h, L - 1, j])
        row.append(s / Hq)
    m = max(row) + 1.0
    per_head = row + [m]
    return torch.tensor([per_head[:] for _ in range(num_kv_heads)], dtype=torch.float32)


def check_example(solve):
    attn = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0], [0.3, 0.3, 0.4, 0.0, 0.0],
         [0.2, 0.2, 0.3, 0.3, 0.0], [0.1, 0.2, 0.3, 0.3, 0.1]],
        [[1.0, 0.0, 0.0, 0.0, 0.0], [0.4, 0.6, 0.0, 0.0, 0.0], [0.2, 0.4, 0.4, 0.0, 0.0],
         [0.1, 0.3, 0.3, 0.3, 0.0], [0.2, 0.2, 0.2, 0.3, 0.1]],
        [[1.0, 0.0, 0.0, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0, 0.0], [0.4, 0.3, 0.3, 0.0, 0.0],
         [0.3, 0.3, 0.2, 0.2, 0.0], [0.15, 0.15, 0.3, 0.3, 0.1]],
        [[1.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0], [0.35, 0.35, 0.3, 0.0, 0.0],
         [0.25, 0.25, 0.2, 0.3, 0.0], [0.1, 0.3, 0.2, 0.3, 0.1]],
    ])
    out = solve(attn, 2)
    exp = torch.tensor([
        [0.1375, 0.2125, 0.25, 0.3, 1.3],
        [0.1375, 0.2125, 0.25, 0.3, 1.3],
    ])
    assert torch.allclose(out, exp, atol=1e-5), out


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(30):
        Hq = int(torch.randint(1, 8, (1,)))
        L = int(torch.randint(2, 12, (1,)))
        H = int(torch.randint(1, Hq + 1, (1,)))
        attn = torch.rand(Hq, L, L)
        # 归一化每行（模拟 softmax + causal 都行）
        attn = attn / attn.sum(dim=-1, keepdim=True)
        got = solve(attn, H)
        exp = _oracle(attn, H)
        assert torch.allclose(got, exp, atol=1e-5), (Hq, L, H, got, exp)


def check_rows_identical(solve):
    # 所有 KV 头的分数完全相同（TOVA 是全局共享）
    attn = torch.rand(4, 6, 6)
    attn = attn / attn.sum(dim=-1, keepdim=True)
    out = solve(attn, 3)
    for i in range(1, 3):
        assert torch.allclose(out[0], out[i]), out


def check_tail_max_plus_one(solve):
    attn = torch.rand(3, 5, 5)
    attn = attn / attn.sum(dim=-1, keepdim=True)
    out = solve(attn, 2)
    body = out[:, :-1]
    tail = out[:, -1]
    expected = float(body.max()) + 1.0
    assert torch.allclose(tail, torch.full_like(tail, expected)), (tail, expected)


check_example.example = True
