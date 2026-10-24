import torch


def _oracle(cands, P, eps, u_bonus):
    """独立纯 Python：前缀去重成树，父链全接受取最深（同深取字典序最小），bonus=CDF 反查。"""
    Pl = P.tolist()
    eps = float(eps)
    nodes = set()
    for c in cands:
        for d in range(1, len(c) + 1):
            nodes.add(tuple(c[:d]))

    def accepted(path):
        for i in range(1, len(path) + 1):
            if Pl[i - 1][path[i - 1]] < eps:
                return False
        return True

    best = []
    for node in sorted(nodes):
        if accepted(node):
            lst = list(node)
            if len(lst) > len(best) or (len(lst) == len(best) and lst < best):
                best = lst
    D = len(best)
    cum = 0.0
    pick = len(Pl[D]) - 1
    for v, r in enumerate(Pl[D]):
        cum += r
        if u_bonus < cum:
            pick = v
            break
    return best + [pick]


def check_example(solve):
    cands = [[1, 2], [1, 3], [0], [1, 2, 0]]
    P = torch.tensor([[0.5, 0.3, 0.1, 0.1],
                      [0.1, 0.3, 0.4, 0.2],
                      [0.3, 0.1, 0.2, 0.4],
                      [0.1, 0.5, 0.3, 0.1]])
    out = solve(cands, P, 0.2, 0.4)
    assert torch.equal(out, torch.tensor([1, 2, 0, 1])), out


def check_tie_break_lexicographic(solve):
    # [1,2] 与 [1,3] 都被接受且同深 -> 取字典序更小的 [1,2]
    cands = [[1, 2], [1, 3]]
    P = torch.tensor([[0.5, 0.3, 0.1, 0.1],
                      [0.1, 0.3, 0.4, 0.2],
                      [0.3, 0.1, 0.2, 0.4],
                      [0.1, 0.5, 0.3, 0.1]])
    out = solve(cands, P, 0.2, 0.05)
    assert torch.equal(out, torch.tensor([1, 2, 0])), out


def check_only_root(solve):
    # eps 抬高到超过所有位置概率 -> 无候选被接受，只吐 P[0] 上的 bonus
    cands = [[0], [1], [1, 2]]
    P = torch.tensor([[0.5, 0.3, 0.1, 0.1],
                      [0.1, 0.3, 0.4, 0.2],
                      [0.3, 0.1, 0.2, 0.4]])
    out = solve(cands, P, 0.6, 0.95)
    assert out.shape[0] == 1, out
    assert torch.equal(out, torch.tensor([3])), out


def check_parent_chain_required(solve):
    # 深层 token 概率达标，但其父 token 不达标 -> 整条路径不被接受
    # [1] P[0][1]=0.3>=0.4? 否 -> 接受链断在根；故只从 P[0] 采样 bonus
    cands = [[1, 0]]
    P = torch.tensor([[0.1, 0.3, 0.3, 0.3],
                      [0.5, 0.1, 0.2, 0.2],
                      [0.25, 0.25, 0.25, 0.25]])
    out = solve(cands, P, 0.4, 0.2)
    assert torch.equal(out, torch.tensor([1])), out


def check_matches_oracle(solve):
    torch.manual_seed(7)
    for _ in range(40):
        V = int(torch.randint(2, 6, (1,)))
        maxd = int(torch.randint(1, 4, (1,)))
        rows = maxd + 1
        raw = torch.rand(rows, V) + 0.05
        P = raw / raw.sum(dim=1, keepdim=True)
        nc = int(torch.randint(1, 5, (1,)))
        cands = []
        for _ in range(nc):
            ln = int(torch.randint(1, maxd + 1, (1,)))
            cands.append([int(x) for x in torch.randint(0, V, (ln,))])
        eps = float(0.05 + torch.rand(1) * 0.5)
        u = float(torch.rand(1) * 0.98)
        out = solve(cands, P, eps, u)
        exp = torch.tensor(_oracle(cands, P, eps, u), dtype=torch.long)
        assert torch.equal(out, exp), (cands, eps, u, out, exp)


check_example.example = True
