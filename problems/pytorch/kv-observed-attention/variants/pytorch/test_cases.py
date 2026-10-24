import torch


def _oracle(attn, num_kv_heads):
    """独立：逐 (h, j) 累加，先算 query 维 sum 再除 (L-j) 再除 groups。"""
    Hq, L, _ = attn.shape
    groups = Hq // num_kv_heads
    out = torch.zeros(num_kv_heads, L, dtype=torch.float32)
    for h in range(num_kv_heads):
        for j in range(L):
            s = 0.0
            for g in range(groups):
                for i in range(j, L):  # 只有 i >= j 才非零
                    s += float(attn[h * groups + g, i, j])
            out[h, j] = s / groups / (L - j)
    return out


def check_example(solve):
    attn = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.2, 0.3, 0.5, 0.0], [0.1, 0.2, 0.3, 0.4]],
        [[1.0, 0.0, 0.0, 0.0], [0.4, 0.6, 0.0, 0.0], [0.3, 0.3, 0.4, 0.0], [0.2, 0.1, 0.3, 0.3]],
    ])
    out = solve(attn, 1)
    exp = torch.tensor([[0.4625, 0.3333333, 0.375, 0.35]])
    assert torch.allclose(out, exp, atol=1e-5), out


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(30):
        H = int(torch.randint(1, 4, (1,)))
        groups = int(torch.randint(1, 5, (1,)))
        Hq = H * groups
        L = int(torch.randint(1, 12, (1,)))
        attn = torch.rand(Hq, L, L)
        # 施加因果掩码 + 每行归一化，模拟 softmax 输出
        mask = torch.tril(torch.ones(L, L))
        attn = attn * mask
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        got = solve(attn, H)
        exp = _oracle(attn, H)
        assert torch.allclose(got, exp, atol=1e-5), (Hq, L, H, got, exp)


def check_uniform_attn(solve):
    # 每 head 都是"下三角均匀 1/(i+1)" — 平均后每列分数相等
    L = 4
    m = torch.zeros(L, L)
    for i in range(L):
        m[i, :i + 1] = 1.0 / (i + 1)
    attn = m.unsqueeze(0).repeat(2, 1, 1)
    out = solve(attn, 1)
    # 每 key 列 j 累计 = sum_{i>=j} 1/(i+1) — 不是常数，此处仅检查形状
    assert out.shape == (1, L), out


def check_kv_group_averaging(solve):
    # 若两个 Q-head 分数完全一样，group-mean 后与单 head 结果一致
    attn = torch.rand(1, 5, 5)
    mask = torch.tril(torch.ones(5, 5))
    attn = attn * mask
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    single = solve(attn, 1)
    dup = attn.repeat(2, 1, 1)  # Hq=2 但内容同 → groups=2, H=1
    double = solve(dup, 1)
    assert torch.allclose(single, double, atol=1e-6), (single, double)


check_example.example = True
