import torch


def _oracle(points, adj, entry, query):
    def dist2(i):
        s = 0.0
        for t in range(points.shape[1]):
            v = float(points[i, t]) - float(query[t])
            s += v * v
        return s
    cur = int(entry)
    while True:
        cd = dist2(cur)
        cand = []
        for nb in adj[cur].tolist():
            nb = int(nb)
            if nb >= 0:
                d = dist2(nb)
                if d < cd:
                    cand.append((d, nb))
        if not cand:
            return torch.tensor(cur, dtype=torch.long)
        cand.sort(key=lambda z: (z[0], z[1]))
        cur = cand[0][1]


CHAIN_P = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])
CHAIN_A = torch.tensor([[1, -1], [0, 2], [1, 3], [2, 4], [3, -1]])
TRAP_P = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [3.0, 3.0], [5.0, 1.0]])
TRAP_A = torch.tensor([[1, 2, -1], [0, 3, -1], [0, 4, -1], [1, 4, -1], [2, 3, -1]])


def check_example(solve):
    y = solve(CHAIN_P, CHAIN_A, 0, torch.tensor([3.7]))
    assert y.item() == 4, f"got {y.item()}"


check_example.example = True


def check_matches_oracle(solve):
    cases = [
        (CHAIN_P, CHAIN_A, 0, torch.tensor([3.7])),
        (CHAIN_P, CHAIN_A, 4, torch.tensor([3.7])),
        (CHAIN_P, CHAIN_A, 2, torch.tensor([-0.4])),
        (TRAP_P, TRAP_A, 0, torch.tensor([4.5, 0.5])),
        (TRAP_P, TRAP_A, 3, torch.tensor([4.5, 0.5])),
        (torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [-10.0, 0.0], [0.0, -10.0]]),
         torch.tensor([[1, 2, 3, 4], [0, -1, -1, -1], [0, -1, -1, -1],
                       [0, -1, -1, -1], [0, -1, -1, -1]]), 1, torch.tensor([0.1, 0.0])),
    ]
    for p, a, e, q in cases:
        y = solve(p, a, e, q)
        assert torch.equal(y, _oracle(p, a, e, q)), (e, y.item())


def check_entry_is_local_optimum(solve):
    # 从链尾 node4 进入，query 就在附近，邻居 node3 更远 -> 立即停在入口
    y = solve(CHAIN_P, CHAIN_A, 4, torch.tensor([3.7]))
    assert y.item() == 4, y.item()


def check_greedy_gets_stuck(solve):
    # 贪心从入口 0 出发停在 1（局部最优），但全局最近其实是 4
    y = solve(TRAP_P, TRAP_A, 0, torch.tensor([4.5, 0.5]))
    assert y.item() == 1, y.item()
    d = ((TRAP_P - torch.tensor([4.5, 0.5])) ** 2).sum(1)
    assert int(d.argmin()) == 4, d.tolist()   # 说明贪心未达全局最优


def check_dtype_and_range(solve):
    y = solve(CHAIN_P, CHAIN_A, 1, torch.tensor([2.3]))
    assert y.dim() == 0 and y.dtype == torch.int64, (y.dim(), y.dtype)
    assert 0 <= int(y) < CHAIN_P.shape[0], int(y)
