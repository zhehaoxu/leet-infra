import torch


def _rank(cands, primary, secondary, tertiary):
    order = cands
    order = order[torch.argsort(tertiary[order], stable=True)]
    order = order[torch.argsort(secondary[order], stable=True)]
    order = order[torch.argsort(primary[order], stable=True)]
    return order


def _oracle(mask, weight, grad, sparsity, eps, u):
    m = mask.flatten()
    n = m.numel()
    w_abs = weight.flatten().abs()
    g_abs = grad.flatten().abs()
    uu = u.flatten()
    idx = torch.arange(n).to(torch.float32)
    act = m > 0
    pool = torch.nonzero(~act).flatten()
    k = n - int(round(sparsity * n))
    react, pruned = [], []
    zero_key = torch.zeros(n)

    na = int(act.sum())
    if na < k:
        grow = _rank(pool, -g_abs, -uu, idx)[: k - na]
        act[grow] = True
        react.append(grow)
        pool = pool[~torch.isin(pool, grow)]
    elif na > k:
        cands = torch.nonzero(act).flatten()
        drop = _rank(cands, w_abs, zero_key, idx)[: na - k]
        act[drop] = False
        pruned.append(drop)

    r = min(int(round(eps * int((~act).sum()))), int(act.sum()), int(pool.numel()))
    if r > 0:
        cands = torch.nonzero(act).flatten()
        drop = _rank(cands, w_abs, zero_key, idx)[:r]
        act[drop] = False
        pruned.append(drop)
        grow = _rank(pool, -g_abs, -uu, idx)[:r]
        act[grow] = True
        react.append(grow)

    new_mask = act.to(torch.float32).reshape(mask.shape)
    join = lambda parts: torch.sort(torch.cat(parts)).values.to(torch.long) if parts else torch.tensor([], dtype=torch.long)
    return new_mask, join(react), join(pruned)


MASK = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
W = torch.tensor([3.0, -1.0, 0.5, 2.0, -0.2, 4.0])
G = torch.tensor([0.1, 2.0, -3.0, 0.4, 5.0, -1.0])
U = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def check_example(solve):
    new_mask, react, pruned = solve(MASK, W, G, 1.0 / 3.0, 0.5, U)
    assert new_mask.tolist() == [1.0, 1.0, 0.0, 1.0, 1.0, 0.0], new_mask.tolist()
    assert react.tolist() == [4], react.tolist()
    assert pruned.tolist() == [2], pruned.tolist()


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(41)
    for n in (8, 12, 20):
        mask = (torch.rand(n) > 0.5).to(torch.float32)
        w = torch.randn(n)
        g = torch.randn(n)
        u = torch.rand(n)
        for sparsity, eps in [(0.25, 0.5), (0.5, 0.3), (0.75, 1.0), (0.5, 0.0)]:
            got = solve(mask, w, g, sparsity, eps, u)
            exp = _oracle(mask, w, g, sparsity, eps, u)
            assert torch.equal(got[0], exp[0]), f"n={n} s={sparsity} eps={eps} mask"
            assert got[1].tolist() == exp[1].tolist(), f"n={n} s={sparsity} eps={eps} react"
            assert got[2].tolist() == exp[2].tolist(), f"n={n} s={sparsity} eps={eps} pruned"


def check_active_count_is_k(solve):
    torch.manual_seed(42)
    for sparsity in (0.2, 0.5, 0.8):
        n = 10
        mask = (torch.rand(n) > 0.5).to(torch.float32)
        new_mask, _, _ = solve(mask, torch.randn(n), torch.randn(n), sparsity, 0.4, torch.rand(n))
        k = n - int(round(sparsity * n))
        assert int(new_mask.sum()) == k, (sparsity, int(new_mask.sum()), k)


def check_eps_zero_only_prunes(solve):
    mask = torch.ones(8)
    w = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.9, 0.8, 0.7])
    g = torch.arange(8).to(torch.float32)
    new_mask, react, pruned = solve(mask, w, g, 0.5, 0.0, torch.zeros(8))
    assert react.numel() == 0, "eps=0 不应重激活"
    assert pruned.tolist() == [4, 5, 6, 7], pruned.tolist()
    assert new_mask.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def check_sparsity_extremes(solve):
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    w = torch.tensor([1.0, 2.0, 3.0, 4.0])
    g = torch.tensor([1.0, 2.0, 3.0, 4.0])
    u = torch.tensor([0.1, 0.2, 0.3, 0.4])
    full, react, pruned = solve(mask, w, g, 0.0, 0.5, u)
    assert full.tolist() == [1.0, 1.0, 1.0, 1.0], "sparsity=0 应全激活"
    assert react.tolist() == [1, 3], react.tolist()
    empty, _, pruned2 = solve(mask, w, g, 1.0, 0.5, u)
    assert empty.tolist() == [0.0, 0.0, 0.0, 0.0], "sparsity=1 应全剪"
    assert pruned2.tolist() == [0, 2], pruned2.tolist()


def check_reactivated_were_inactive(solve):
    torch.manual_seed(43)
    n = 16
    mask = (torch.rand(n) > 0.5).to(torch.float32)
    new_mask, react, pruned = solve(mask, torch.randn(n), torch.randn(n), 0.5, 0.6, torch.rand(n))
    old = set(torch.nonzero(mask.flatten() == 0).flatten().tolist())
    assert set(react.tolist()) <= old, "重激活元素必须原本是非激活"
    assert not (set(react.tolist()) & set(pruned.tolist())), "重激活与剪枝集合不相交"
    for i in pruned.tolist():
        assert mask.flatten()[i] > 0, "被剪元素必须原本是激活的"


def check_tie_break_by_u(solve):
    # 两个非激活元素 |grad| 相同：u 更大的优先重激活
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    w = torch.tensor([-1.0, -2.0, 0.5, 0.4])
    g = torch.tensor([0.0, 0.0, 1.0, 1.0])
    u = torch.tensor([0.0, 0.0, 0.2, 0.9])
    new_mask, react, _ = solve(mask, w, g, 0.5, 0.5, u)
    assert react.tolist() == [3], react.tolist()
    assert new_mask.tolist() == [0.0, 1.0, 0.0, 1.0], new_mask.tolist()


def check_output_dtypes_and_shape(solve):
    mask = torch.zeros(2, 3)
    mask[0, 0] = 1.0
    mask[1, 2] = 1.0
    w = torch.randn(2, 3)
    g = torch.randn(2, 3)
    u = torch.rand(2, 3)
    new_mask, react, pruned = solve(mask, w, g, 0.5, 0.5, u)
    assert new_mask.shape == (2, 3) and new_mask.dtype == torch.float32
    assert react.dtype == torch.long and pruned.dtype == torch.long
    assert set(new_mask.flatten().tolist()) <= {0.0, 1.0}
