import torch


def _rank_desc(row, n):
    order = sorted(range(len(row)), key=lambda j: (-row[j], j))
    return order[:n]


def _oracle(attn, k, p):
    """逐步 python 重算：手写排序与累加，结构异于参考解的 argsort 路径。"""
    steps, length = attn.shape
    prompt = length - steps
    rows = [[float(x) for x in attn[s]] for s in range(steps)]
    ns, keep = [], [[False] * length for _ in range(steps)]
    prev = None
    for s in range(steps):
        valid = prompt + s
        if valid <= 0:
            ns.append(0)
            prev = []
            continue
        if prev is None:
            cand = _rank_desc(rows[s][:valid], k)
        else:
            ordered = sorted(prev, key=lambda j: (-rows[s][j], j))
            cand = ordered
        total = sum(rows[s][:valid])
        cum, n = 0.0, len(cand)
        for i, j in enumerate(cand, start=1):
            cum += rows[s][j]
            if cum >= p * total - 1e-12:
                n = i
                break
        for j in cand[:n]:
            keep[s][j] = True
        ns.append(n)
        prev = _rank_desc(rows[s][:valid], k)
    return torch.tensor(ns, dtype=torch.long), torch.tensor(keep, dtype=torch.bool)


def check_example(solve):
    torch.manual_seed(0)
    attn = torch.zeros(3, 6)
    attn[0, :3] = torch.tensor([0.5, 0.3, 0.2])
    attn[1, :4] = torch.tensor([0.5, 0.3, 0.1, 0.1])
    attn[2, :5] = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
    n, keep = solve(attn, 3, 0.75)
    assert n.tolist() == [2, 2, 3], f"got {n.tolist()}"
    assert keep[0].tolist() == [True, True, False, False, False, False], f"got {keep[0].tolist()}"
    assert keep[1].tolist() == [True, True, False, False, False, False], f"got {keep[1].tolist()}"
    assert keep[2].tolist() == [True, True, True, False, False, False], f"got {keep[2].tolist()}"


check_example.example = True


def check_pattern_reuse_limits_recall(solve):
    """候选只能来自上一步 top-k：当步在候选外的高分 token 不会被选中（近似性的代价）。"""
    torch.manual_seed(1)
    attn = torch.zeros(2, 5)
    attn[0, :3] = torch.tensor([0.8, 0.1, 0.1])      # 上一步 top2 = {0,1}
    attn[1, :4] = torch.tensor([0.05, 0.05, 0.85, 0.05])
    n, keep = solve(attn, 2, 0.5)
    assert not bool(keep[1, 2]), "token2 不在上一步 top-2 内，即使当步最高也不应入选"
    assert keep[1, :2].all(), "候选集内按当步分数排序后应保留满足质量的前若干"
    assert int(n[1]) == 2


def check_p_controls_budget(solve):
    """p 越大 n 越大（单一调参方向）：同一输入 p=0.2 的 n 不超过 p=0.9 的 n。"""
    torch.manual_seed(2)
    attn = torch.zeros(1, 4)
    attn[0, :4] = torch.tensor([0.4, 0.3, 0.2, 0.1])
    low, _ = solve(attn, 4, 0.2)
    high, _ = solve(attn, 4, 0.9)
    assert int(low[0]) <= int(high[0]), f"p 更大不应得到更小的 n：{int(low[0])} vs {int(high[0])}"
    zero, _ = solve(attn, 4, 0.0)
    assert int(zero[0]) == 1, "p=0 时最小预算应为 1"


def check_capped_by_k_and_valid(solve):
    """n 不超过候选数，候选数不超过 min(k, 有效列)：早期步 k 大于有效列时自动收缩。"""
    torch.manual_seed(3)
    attn = torch.zeros(2, 5)
    attn[0, :3] = torch.tensor([0.2, 0.2, 0.2])
    attn[1, :4] = torch.tensor([0.2, 0.2, 0.2, 0.2])
    n, keep = solve(attn, 9, 0.99)
    assert int(n[0]) == 3 and int(n[1]) == 3, f"got {n.tolist()}"
    assert int(keep[0].sum()) == 3 and int(keep[1].sum()) == 3


def check_matches_oracle_and_errors(solve):
    """随机用例与逐步重算对拍，并校验非法参数与全零步。"""
    torch.manual_seed(21)
    attn = torch.rand(5, 10)
    for s in range(5):
        attn[s, 5 + s:] = 0.0
    for k, p in ((3, 0.6), (7, 0.95), (1, 0.0)):
        exp_n, exp_k = _oracle(attn, k, p)
        n, kp = solve(attn, k, p)
        assert torch.equal(n, exp_n), f"k={k},p={p} 预算不符：{n.tolist()} vs {exp_n.tolist()}"
        assert torch.equal(kp, exp_k), f"k={k},p={p} 掩码不符"
    zero = torch.zeros(2, 6)
    n0, k0 = solve(zero, 3, 0.5)
    assert int(n0[0]) == 1 and int(k0.sum()) == 2, "全零步应退化为每步 1 个"
    for bad in ((attn, 0, 0.5), (attn, 2, 1.5)):
        try:
            solve(*bad)
            raise AssertionError(f"{bad[1:]} 应抛 ValueError")
        except ValueError:
            pass
