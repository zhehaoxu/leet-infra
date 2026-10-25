import heapq
import math
import random

import torch


def _oracle(parent, acc, k, gmax):
    """独立重算（与参考解结构不同）：
    1) 逐节点沿 parent 链显式上溯、自顶向下逐元素乘 acc 得到 V 与深度（不做拓扑序 DP 复用）；
    2) 把全部节点按深度分桶；
    3) 自浅到深逐桶筛出「父节点已入选（或父为根）」的候选，用 heapq.nlargest 按 (V, -index)
       取走剩余预算，而不是 sort 后切片；期望值用 math.fsum 另序求和。
    这里必须保持「逐层把预算填满」的贪心语义，不能改成对全树取全局 top-k：两者在稀疏层上
    给出的点集不同（见 check_sparse_level_greedy_not_global_topk）。"""
    pl = [int(x) for x in parent.tolist()]
    al = [float(x) for x in acc.tolist()]
    M = len(pl)
    info = {}

    def chain(v):
        if v in info:
            return info[v]
        cur, seen = v, []
        while cur != 0:
            seen.append(cur)
            cur = pl[cur]
        prod = 1.0
        for u in reversed(seen):
            prod *= al[u]
        info[v] = (prod, len(seen))
        return info[v]

    V, buckets = {}, {}
    for v in range(1, M):
        prod, dd = chain(v)
        V[v] = prod
        buckets.setdefault(dd, []).append(v)

    sel, used = set(), 0
    for dd in sorted(buckets):
        if used >= k or dd > gmax:
            break
        cands = [v for v in buckets[dd] if pl[v] == 0 or pl[v] in sel]
        if not cands:
            break
        keep = heapq.nlargest(k - used, cands, key=lambda v: (V[v], -v))
        sel.update(keep)
        used += len(keep)
    return sorted(sel), 1.0 + math.fsum(V[v] for v in sorted(sel))


def _run(solve, parent, acc, k, gmax):
    sel, E = solve(parent, acc, k, gmax)
    exp_sel, exp_E = _oracle(parent, acc, k, gmax)
    assert torch.equal(sel, torch.tensor(exp_sel, dtype=torch.long)), (k, gmax, sel.tolist(), exp_sel)
    assert torch.allclose(torch.tensor([E]), torch.tensor([exp_E]), atol=1e-5, rtol=1e-5), (k, gmax, E, exp_E)
    return sel.tolist(), E


def check_example(solve):
    parent = torch.tensor([-1, 0, 0, 1, 1, 2, 4])
    acc = torch.tensor([0.0, 0.75, 0.5, 0.875, 0.625, 0.5, 0.25])
    sel, E = solve(parent, acc, 4, 3)
    assert torch.equal(sel, torch.tensor([1, 2, 3, 4])), sel
    assert abs(E - 3.375) < 1e-5, E
    sel, E = solve(parent, acc, 1, 3)
    assert torch.equal(sel, torch.tensor([1])), sel
    assert abs(E - 1.75) < 1e-5, E
    sel, E = solve(parent, acc, 4, 1)
    assert torch.equal(sel, torch.tensor([1, 2])), sel
    assert abs(E - 2.25) < 1e-5, E
    sel, E = solve(parent, acc, 0, 3)
    assert sel.numel() == 0 and abs(E - 1.0) < 1e-9, (sel, E)
    _run(solve, parent, acc, 4, 3)


def check_matches_oracle(solve):
    rng = random.Random(240616858)
    for _ in range(300):
        M = rng.randint(1, 40)
        parent = [-1] + [0 if i == 1 else rng.randrange(i) for i in range(1, M)]
        acc = [0.0] + [round(rng.uniform(0.0, 1.0), 3) for _ in range(M - 1)]
        k = rng.randint(0, 12)
        gmax = rng.randint(0, 6)
        _run(solve, torch.tensor(parent, dtype=torch.long), torch.tensor(acc), k, gmax)


def check_sparse_level_greedy_not_global_topk(solve):
    # 逐层贪心在稀疏的第一层会把预算摊给整层：V=(0.9,0.5,0.63,0.54,0.4,0.216) 时
    # k=3 选 {1,2,3}（Σ=2.03），而「全局 top-3」会选 {1,3,4}（Σ=2.07）——题面规定的是前者。
    parent = torch.tensor([-1, 0, 0, 1, 1, 2, 4])
    acc = torch.tensor([0.0, 0.9, 0.5, 0.7, 0.6, 0.8, 0.4])
    sel, E = _run(solve, parent, acc, 3, 3)
    assert sel == [1, 2, 3], sel
    assert abs(E - 3.03) < 1e-4, E


def check_zero_budget_and_zero_levels(solve):
    parent = torch.tensor([-1, 0, 1, 2])
    acc = torch.tensor([0.0, 0.9, 0.8, 0.7])
    for k, gmax in [(0, 3), (5, 0), (0, 0)]:
        sel, E = solve(parent, acc, k, gmax)
        assert sel.numel() == 0, (k, gmax, sel)
        assert abs(E - 1.0) < 1e-9, (k, gmax, E)
        _run(solve, parent, acc, k, gmax)


def check_chain_closed_form(solve):
    # 链状树（每点恰一个孩子）：贪心就是自根往下取前 min(k, gmax, 链长) 个，V 是 acc 的前缀积
    M = 7
    parent = torch.tensor([-1] + list(range(M - 1)))
    acc = torch.tensor([0.0] + [0.8] * (M - 1))
    for k, gmax in [(3, 6), (10, 6), (10, 2), (6, 6), (2, 0)]:
        n = min(k, gmax, M - 1)
        exp_sel = list(range(1, n + 1))
        prefix = 1.0
        exp_E = 1.0
        for i in range(n):
            prefix *= 0.8
            exp_E += prefix
        sel, E = solve(parent, acc, k, gmax)
        assert torch.equal(sel, torch.tensor(exp_sel, dtype=torch.long)), (k, gmax, sel)
        assert abs(E - exp_E) < 1e-6, (k, gmax, E, exp_E)


def check_all_accept_and_ancestor_closure(solve):
    # acc 全 1 -> V 恒为 1，选点变成「按层、层内按编号」的广度填充，E = 1 + |sel|
    rng = random.Random(7)
    for _ in range(40):
        M = rng.randint(2, 25)
        parent = [-1] + [rng.randrange(i) for i in range(1, M)]
        acc = torch.ones(M)
        k = rng.randint(1, 10)
        gmax = rng.randint(1, 5)
        pt = torch.tensor(parent, dtype=torch.long)
        sel, E = solve(pt, acc, k, gmax)
        assert sel.dtype == torch.long
        assert abs(E - (1.0 + sel.numel())) < 1e-6, (sel.tolist(), E)
        sel_list = sel.tolist()
        assert sel_list == sorted(sel_list) and len(set(sel_list)) == len(sel_list)
        # 祖先封闭 + 深度不超过 gmax
        depth = {}

        def dep(v):
            if v in depth:
                return depth[v]
            d = 1 if parent[v] == 0 else dep(parent[v]) + 1
            depth[v] = d
            return d

        for v in sel_list:
            assert v != 0
            assert dep(v) <= gmax, (v, gmax)
            if parent[v] != 0:
                assert parent[v] in sel_list, (v, parent[v], sel_list)
        _run(solve, pt, acc, k, gmax)


def check_frozen_branch_is_dropped(solve):
    # 第一层没被留下的节点，其整棵子树永远不进 sel：k=2 时节点 3 与其子 6 一起被剪掉。
    # V = (0.4, 0.3, 0.2, 0.4, 0.3, 0.2)
    parent = torch.tensor([-1, 0, 0, 0, 1, 2, 3])
    acc = torch.tensor([0.0, 0.4, 0.3, 0.2, 1.0, 1.0, 1.0])
    sel, E = _run(solve, parent, acc, 2, 3)
    assert sel == [1, 2] and abs(E - 1.7) < 1e-6, (sel, E)
    sel, E = _run(solve, parent, acc, 3, 3)
    assert sel == [1, 2, 3] and abs(E - 1.9) < 1e-6, (sel, E)
    sel, E = _run(solve, parent, acc, 4, 3)
    assert sel == [1, 2, 3, 4] and abs(E - 2.3) < 1e-6, (sel, E)
    # 预算给到 5，第二层才按分数高低补回两个孩子
    sel, E = _run(solve, parent, acc, 5, 3)
    assert sel == [1, 2, 3, 4, 5] and abs(E - 2.6) < 1e-6, (sel, E)


check_example.example = True
