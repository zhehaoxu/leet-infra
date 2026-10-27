import itertools

import torch


def _oracle(cand_tokens, cand_lps, budget, prefix_len):
    """穷举全部路径 + 全局排序取前 budget 条（结构异于参考解的堆式 best-first）。"""
    heads, k = cand_tokens.shape
    lps = cand_lps.numpy().astype(float)
    scored = []
    for depth in range(1, heads + 1):
        for path in itertools.product(range(k), repeat=depth):
            score = sum(lps[d, path[d]] for d in range(depth))
            scored.append((-score, path))
    scored.sort(key=lambda item: (item[0], item[1]))
    chosen = [path for _, path in scored[:budget]]

    index = {(): -1}
    for i, path in enumerate(chosen):
        index[path] = i
    depth = [len(path) for path in chosen]
    parent = [index[path[:-1]] for path in chosen]
    tokens = [int(cand_tokens[len(path) - 1, path[-1]]) for path in chosen]
    mask = torch.zeros(len(chosen), len(chosen) + prefix_len, dtype=torch.bool)
    mask[:, :prefix_len] = True
    for i, path in enumerate(chosen):
        for d in range(1, len(path) + 1):
            mask[i, prefix_len + index[path[:d]]] = True
    return (torch.tensor(parent, dtype=torch.int64),
            torch.tensor(tokens, dtype=torch.int64),
            torch.tensor(depth, dtype=torch.int64),
            torch.tensor([prefix_len + d - 1 for d in depth], dtype=torch.int64),
            mask)


def _tokens(*rows):
    return torch.tensor(rows, dtype=torch.int64)


def _lps(*rows):
    return torch.tensor(rows, dtype=torch.float32)


def check_example(solve):
    torch.manual_seed(0)
    toks = _tokens([7, 9], [3, 8])
    lps = _lps([-0.1, -0.5], [-0.2, -3.0])
    parent, tokens, depth, positions, mask = solve(toks, lps, 4, 3)

    assert parent.tolist() == [-1, 0, -1, 2], f"got {parent.tolist()}"
    assert tokens.tolist() == [7, 3, 9, 3], f"got {tokens.tolist()}"
    assert depth.tolist() == [1, 2, 1, 2], f"got {depth.tolist()}"
    assert positions.tolist() == [3, 4, 3, 4], f"got {positions.tolist()}"

    want = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 1, 1],
    ], dtype=torch.bool)
    assert torch.equal(mask, want), f"mask 不符\n{mask.to(torch.int64)}"


check_example.example = True


def check_uniform_gives_level_order(solve):
    """所有 logprob 相同时（都 -1），候选按深度分层出队，且与全局排序一致。"""
    torch.manual_seed(1)
    heads, k, budget = 3, 2, 5
    toks = _tokens([10, 11], [20, 21], [30, 31])
    lps = _lps(*[[-1.0] * k for _ in range(heads)])
    got = solve(toks, lps, budget, 2)
    exp = _oracle(toks, lps, budget, 2)
    for name, a, b in zip(("parent", "tokens", "depth", "positions", "mask"), got, exp):
        assert torch.equal(a, b), f"{name} 不符：{a.tolist()} vs {b.tolist()}"
    assert got[2].tolist() == [1, 1, 2, 2, 2], f"应分层出队，got {got[2].tolist()}"
    assert got[1].tolist() == [10, 11, 20, 21, 20], f"got {got[1].tolist()}"


def check_mask_positions_and_prefix_closure(solve):
    """随机树上校验：前缀封闭、父下标小于子下标、同深度共享 position、兄弟互不可见。"""
    torch.manual_seed(2)
    heads, k, budget, prefix_len = 3, 3, 8, 4
    toks = torch.randint(0, 50, (heads, k))
    lps = -torch.rand(heads, k, dtype=torch.float32)
    got = solve(toks, lps, budget, prefix_len)
    exp = _oracle(toks, lps, budget, prefix_len)
    for name, a, b in zip(("parent", "tokens", "depth", "positions", "mask"), got, exp):
        assert torch.equal(a, b), f"{name} 不符：{a.tolist()} vs {b.tolist()}"

    parent, _, depth, positions, mask = got
    n = parent.numel()
    assert n == budget, f"应正好吐出 budget 个节点，got {n}"
    for i in range(n):
        p = int(parent[i])
        assert p < i, f"父下标必须小于子下标：node{i} <- {p}"
        if p >= 0:
            assert int(depth[p]) + 1 == int(depth[i]), "子节点深度应为父节点 + 1"
    assert torch.equal(positions, prefix_len + depth - 1), "position_ids 应为 prefix_len+depth-1"
    assert int(torch.unique(positions).numel()) == int(depth.max()), "同深度必须共享 position"
    assert bool(mask[:, :prefix_len].all()), "已确认前缀必须全部可见"
    assert bool(mask[torch.arange(n), prefix_len + torch.arange(n)].all()), "节点必须看见自己"
    for i in range(n):
        for j in range(n):
            visible = i == j or _is_ancestor(got, i, j)
            assert bool(mask[i, prefix_len + j]) == visible, (
                f"node{i} 对 node{j} 的可见性不符（兄弟必须互不可见）")


def _is_ancestor(got, i, j):
    parent = got[0]
    cur = int(parent[i])
    while cur >= 0:
        if cur == j:
            return True
        cur = int(parent[cur])
    return False


def check_budget_and_path_enumeration(solve):
    """budget 极小只剩最优单点；给足 budget 则穷尽全部 K + K^2 + ... + K^H 条路径。"""
    torch.manual_seed(3)
    heads, k, prefix_len = 2, 3, 5
    toks = _tokens([4, 5, 6], [7, 8, 9])
    lps = _lps([-0.2, -0.2, -1.0], [-0.1, -1.0, -2.0])

    parent, tokens, depth, positions, mask = solve(toks, lps, 1, prefix_len)
    assert tokens.tolist() == [4], f"最优单点应为 head0 的 4，got {tokens.tolist()}"
    assert parent.tolist() == [-1] and depth.tolist() == [1]
    assert positions.tolist() == [prefix_len]
    assert torch.equal(mask, torch.ones(1, prefix_len + 1, dtype=torch.bool))

    total = sum(k ** d for d in range(1, heads + 1))
    got = solve(toks, lps, total + 5, prefix_len)
    assert got[0].numel() == total, f"应穷尽 {total} 条路径，got {got[0].numel()}"
    exp = _oracle(toks, lps, total + 5, prefix_len)
    for name, a, b in zip(("parent", "tokens", "depth", "positions", "mask"), got, exp):
        assert torch.equal(a, b), f"{name} 不符"


def check_matches_oracle_and_errors(solve):
    """多组随机形状对拍 + 非法输入。"""
    torch.manual_seed(21)
    for heads, k, budget, prefix_len in ((4, 2, 6, 1), (2, 4, 5, 7), (1, 3, 2, 3)):
        toks = torch.randint(0, 100, (heads, k))
        lps = -torch.rand(heads, k, dtype=torch.float32) * 3.0
        got = solve(toks, lps, budget, prefix_len)
        exp = _oracle(toks, lps, budget, prefix_len)
        for name, a, b in zip(("parent", "tokens", "depth", "positions", "mask"), got, exp):
            assert torch.equal(a, b), f"{(heads, k, budget, prefix_len)} {name} 不符"

    toks = _tokens([1, 2], [3, 4])
    good = _lps([-0.1, -0.2], [-0.3, -0.4])
    bad_calls = [
        (toks, _lps([0.1, -0.2], [-0.3, -0.4]), 4, 2),      # logprob > 0
        (toks, good, 0, 2),                                  # budget < 1
        (toks, good, 4, 0),                                  # prefix_len < 1
        (toks, _lps([-0.1, -0.2]), 4, 2),                    # 形状不一致
        (torch.zeros(4, dtype=torch.int64), good, 4, 2),      # 非二维
    ]
    for args in bad_calls:
        try:
            solve(*args)
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
