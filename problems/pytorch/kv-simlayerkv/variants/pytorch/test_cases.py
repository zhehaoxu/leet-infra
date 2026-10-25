import torch

LA = torch.tensor([
    [0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.2, 0.1],
    [0.1, 0.05, 0.2, 0.1, 0.2, 0.1, 0.05, 0.1, 0.05, 0.05],
])


def _oracle(last_attn, n_initial, n_recent, n_last, thr):
    """独立纯 Python oracle：逐层集合构造掩码（与 solution 的张量切片路径不同）。"""
    N, L = last_attn.shape
    disabled = (thr == 1.0) or (L <= n_initial + n_recent + n_last)
    masks, ratios = [], []
    for l in range(N):
        row = [float(x) for x in last_attn[l]]
        if disabled:
            masks.append([True] * L)
            ratios.append(0.0)
            continue
        s = sum(row[:min(n_initial, L)]) + sum(row[max(0, L - n_recent):])
        if s > thr:
            keep = set(range(min(n_initial, L))) | set(range(max(0, L - (n_recent - n_last)), L))
            masks.append([j in keep for j in range(L)])
            ratios.append(float(L - n_initial - (n_recent - n_last)) / float(L))
        else:
            masks.append([True] * L)
            ratios.append(0.0)
    return torch.tensor(masks, dtype=torch.bool), torch.tensor(ratios, dtype=torch.float32)


def check_example(solve):
    mask, ratios = solve(LA, 2, 4, 1, 0.5)
    exp_mask = torch.tensor([
        [True, True, False, False, False, False, False, True, True, True],
        [True, True, True, True, True, True, True, True, True, True],
    ])
    assert torch.equal(mask, exp_mask), mask
    assert torch.allclose(ratios, torch.tensor([0.5, 0.0]), atol=1e-5, rtol=1e-5), ratios


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(21)
    for _ in range(30):
        N = int(torch.randint(1, 5, (1,)))
        n_initial = int(torch.randint(1, 4, (1,)))
        n_last = int(torch.randint(1, 3, (1,)))
        n_recent = n_last + int(torch.randint(1, 7, (1,)))
        L = int(torch.randint(1, 32, (1,)))
        thr = float(torch.rand(1))
        last_attn = torch.rand(N, L)
        last_attn = last_attn / last_attn.sum(dim=1, keepdim=True)
        gm, gr = solve(last_attn, n_initial, n_recent, n_last, thr)
        em, er = _oracle(last_attn, n_initial, n_recent, n_last, thr)
        assert torch.equal(gm, em), (N, L, n_initial, n_recent, n_last, thr, gm, em)
        assert torch.allclose(gr, er, atol=1e-5, rtol=1e-5), (gr, er)


def check_threshold_one_disables(solve):
    # 全部质量堆在头尾（最懒惰的极端），但 θ=1.0 是硬开关：必须原样保留
    delta = torch.zeros(1, 10)
    delta[0, 0], delta[0, 1], delta[0, 9] = 0.5, 0.3, 0.2
    mask, ratios = solve(delta, 2, 4, 1, 1.0)
    assert bool(mask.all()), mask
    assert torch.allclose(ratios, torch.zeros(1), atol=1e-5, rtol=1e-5), ratios


def check_short_sequence_no_compression(solve):
    # L == n_initial + n_recent + n_last → 禁用分支（须严格大于该和才压缩）
    la = torch.ones(2, 7) / 7.0
    mask, ratios = solve(la, 2, 4, 1, 0.0)
    assert bool(mask.all()), mask
    assert torch.allclose(ratios, torch.zeros(2), atol=1e-5, rtol=1e-5), ratios


def check_boundary_strict_greater(solve):
    # 二进制精确值使 s = 0.75 == θ：严格大于 → 不懒；θ 略降 → 懒
    row = torch.tensor([0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.125, 0.125, 0.0])
    m1, r1 = solve(row[None], 2, 4, 1, 0.75)
    assert bool(m1.all()) and torch.allclose(r1, torch.zeros(1), atol=1e-5, rtol=1e-5), (m1, r1)
    m2, r2 = solve(row[None], 2, 4, 1, 0.749)
    exp = torch.tensor([[True, True, False, False, False, False, False, True, True, True]])
    assert torch.equal(m2, exp), m2
    assert torch.allclose(r2, torch.tensor([0.5]), atol=1e-5, rtol=1e-5), r2


def check_zero_attention_not_lazy(solve):
    # 全零注意力：s=0，任意 θ>0 不懒；θ=0 也不懒（严格大于）
    la = torch.zeros(3, 12)
    mask, ratios = solve(la, 2, 4, 1, 0.0)
    assert bool(mask.all()), mask
    assert torch.allclose(ratios, torch.zeros(3), atol=1e-5, rtol=1e-5), ratios
