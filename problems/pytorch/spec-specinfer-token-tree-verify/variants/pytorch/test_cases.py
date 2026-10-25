import random

import torch


def _oracle(parent, token, qdist, T, u, u_final):
    """独立重算：完全不做「逐层前缀共享」的 BFS，而是把每个节点对应的根->节点路径单独取出来，
    对每条路径各自跑一遍线性拒绝采样（重复计算父节点），得到「路径无拒绝」的节点集合；
    再按 (深度, token 字典序, 编号) 选终点；收尾 token 用显式循环重算残差/累进 CDF。"""
    pl = [int(x) for x in parent.tolist()]
    tk = [int(x) for x in token.tolist()]
    Q = qdist.tolist()
    P = T.tolist()
    Ul = u.tolist()
    M = len(pl)
    V = len(P[0]) if M else 0

    kids = [[] for _ in range(M)]
    for v in range(1, M):
        kids[pl[v]].append(v)

    def path_of(v):
        seq = []
        cur = v
        while cur != 0:
            seq.append(cur)
            cur = pl[cur]
        seq.reverse()
        return seq  # 不含根

    def path_survives(seq):
        """逐节点独立判定：任一节点被拒则整条路径作废。"""
        for w in seq:
            par = pl[w]
            q = Q[w][tk[w]]
            p = P[par][tk[w]]
            ratio = p / q if q > 0 else float("inf")
            if not (Ul[w] < min(1.0, ratio)):
                return False
        return True

    def cdf(row, uu):
        c = 0.0
        for i in range(V):
            c = c + row[i]
            if c > uu:
                return i
        return V - 1

    alive = [v for v in range(M) if path_survives(path_of(v))]
    alive.sort(key=lambda v: (-len(path_of(v)), [tk[w] for w in path_of(v)], v))
    last = alive[0]
    seq = path_of(last)
    out = [tk[w] for w in seq]

    bad = [c for c in kids[last] if not path_survives(path_of(c))]
    if bad:
        c0 = min(bad)
        mass = []
        tot = 0.0
        for a in range(V):
            r = P[last][a] - Q[c0][a]
            if r < 0.0:
                r = 0.0
            mass.append(r)
            tot += r
        row = [x / tot for x in mass]
    else:
        row = P[last]
    out.append(cdf(row, float(u_final)))
    return out, seq


def _run(solve, parent, token, qdist, T, u, u_final):
    out, nodes = solve(parent, token, qdist, T, u, u_final)
    exp_out, exp_nodes = _oracle(parent, token, qdist, T, u, u_final)
    assert torch.equal(out, torch.tensor(exp_out, dtype=torch.long)), (out.tolist(), exp_out)
    assert torch.equal(nodes, torch.tensor(exp_nodes, dtype=torch.long)), (nodes.tolist(), exp_nodes)
    assert out.dtype == torch.long and nodes.dtype == torch.long
    assert out.numel() == nodes.numel() + 1, (out.numel(), nodes.numel())
    return out.tolist(), nodes.tolist()


def _mk_tree(rng, M, max_par_lookback=6):
    parent = [-1]
    for v in range(1, M):
        parent.append(rng.randrange(max(1, v - max_par_lookback), v) if v > 1 and rng.random() < 0.7 else 0)
    return parent


def check_example(solve):
    parent = torch.tensor([-1, 0, 0, 1, 1, 3])
    token = torch.tensor([0, 1, 2, 0, 3, 1])
    T = torch.tensor([[0.1, 0.6, 0.2, 0.1],
                      [0.5, 0.1, 0.3, 0.1],
                      [0.2, 0.2, 0.3, 0.3],
                      [0.1, 0.1, 0.6, 0.2],
                      [0.25, 0.25, 0.25, 0.25],
                      [0.4, 0.2, 0.2, 0.2]])
    q = torch.tensor([[0.25, 0.25, 0.25, 0.25],
                      [0.1, 0.3, 0.3, 0.3],
                      [0.2, 0.2, 0.6, 0.0],
                      [0.4, 0.2, 0.2, 0.2],
                      [0.1, 0.1, 0.1, 0.7],
                      [0.3, 0.3, 0.3, 0.1]])
    u = torch.tensor([0.0, 0.5, 0.9, 0.3, 0.5, 0.7])
    out, nodes = solve(parent, token, q, T, u, 0.8)
    assert torch.equal(out, torch.tensor([1, 0, 3])), out
    assert torch.equal(nodes, torch.tensor([1, 3])), nodes
    out, _ = solve(parent, token, q, T, u, 0.5)
    assert torch.equal(out, torch.tensor([1, 0, 2])), out
    _run(solve, parent, token, q, T, u, 0.8)

    p2 = torch.tensor([-1, 0, 1])
    t2 = torch.tensor([0, 2, 0])
    T2 = torch.tensor([[0.1, 0.1, 0.7, 0.1], [0.25, 0.25, 0.25, 0.25], [0.05, 0.15, 0.6, 0.2]])
    q2 = torch.tensor([[0.25] * 4, [0.2, 0.2, 0.3, 0.3], [0.4, 0.2, 0.2, 0.2]])
    u2 = torch.tensor([0.0, 0.4, 0.5])
    out, nodes = solve(p2, t2, q2, T2, u2, 0.3)
    assert torch.equal(out, torch.tensor([2, 0, 2])), out
    assert torch.equal(nodes, torch.tensor([1, 2])), nodes
    _run(solve, p2, t2, q2, T2, u2, 0.3)


def check_matches_oracle(solve):
    rng = random.Random(230509781)
    for _ in range(250):
        M = rng.randint(2, 22)
        V = rng.randint(2, 5)
        parent = _mk_tree(rng, M)
        token = [rng.randrange(V) for _ in range(M)]
        Q, P = [], []
        for _ in range(M):
            rq = [round(rng.uniform(0.0, 1.0), 3) for _ in range(V)]
            rp = [round(rng.uniform(0.0, 1.0), 3) for _ in range(V)]
            sq, sp = sum(rq), sum(rp)
            Q.append([x / sq for x in rq])
            P.append([x / sp for x in rp])
        u = [round(rng.uniform(0.0, 0.999), 4) for _ in range(M)]
        uf = round(rng.uniform(0.0, 0.999), 4)
        _run(solve,
             torch.tensor(parent, dtype=torch.long),
             torch.tensor(token, dtype=torch.long),
             torch.tensor(Q), torch.tensor(P), torch.tensor(u), uf)


def check_root_only_tree(solve):
    # 树上没有任何草稿节点：输出只有 target 在根上的那一个采样
    parent = torch.tensor([-1])
    token = torch.tensor([0])
    T = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    q = torch.tensor([[0.25] * 4])
    out, nodes = solve(parent, token, q, T, torch.tensor([0.0]), 0.35)
    assert torch.equal(out, torch.tensor([2])), out
    assert nodes.numel() == 0, nodes
    _run(solve, parent, token, q, T, torch.tensor([0.0]), 0.35)


def check_first_layer_all_rejected(solve):
    # 第一层全拒：存活路径为空，收尾 token 取自「编号最小的被拒孩子」的残差分布
    parent = torch.tensor([-1, 0, 0, 1])
    token = torch.tensor([0, 0, 1, 2])
    T = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.25] * 4, [0.25] * 4])
    q = torch.tensor([[0.25] * 4,
                      [0.4, 0.3, 0.2, 0.1],   # q_1 = 0.4 > p_1 = T[0][0] = 0.1 -> ratio 0.25
                      [0.3, 0.4, 0.2, 0.1],   # q_2 = 0.4 > p_2 = T[0][1] = 0.2 -> ratio 0.5
                      [0.25] * 4])
    u = torch.tensor([0.0, 0.9, 0.9, 0.0])    # 0 是根，1/2 都被拒
    out, nodes = solve(parent, token, q, T, u, 0.5)
    assert nodes.numel() == 0 and out.numel() == 1, (out, nodes)
    # 残差 = norm(max(T[0]-q[1],0)) = norm([0,0,0.1,0.3]) = [0,0,0.25,0.75] -> CDF 首个 >0.5 是下标 3
    assert torch.equal(out, torch.tensor([3])), out
    _run(solve, parent, token, q, T, u, 0.5)


def check_identical_draft_always_accepts_and_gives_bonus(solve):
    # qdist 的行与父节点对应的 T 行完全相同 -> p==q、ratio=1，u in [0,1) 恒接受，树耗尽后走 bonus
    parent = torch.tensor([-1, 0, 1, 1, 2])
    token = torch.tensor([0, 1, 2, 1, 0])
    T = torch.tensor([[0.2, 0.5, 0.3],
                      [0.4, 0.1, 0.5],
                      [0.1, 0.3, 0.6],
                      [0.3, 0.3, 0.4],
                      [0.5, 0.2, 0.3]])
    # 每个孩子的 q 直接取父节点的 T 行：孩子位置上的 target/draft 完全一致
    q = torch.tensor([[0.2, 0.5, 0.3],
                      [0.2, 0.5, 0.3],   # child of root -> T[0]
                      [0.4, 0.1, 0.5],   # child of 1   -> T[1]
                      [0.4, 0.1, 0.5],   # child of 1   -> T[1]
                      [0.1, 0.3, 0.6]])  # child of 2   -> T[2]
    u = torch.tensor([0.0, 0.999, 0.5, 0.7, 0.999])
    out, nodes = solve(parent, token, q, T, u, 0.25)
    # 全部接受：最深路径 = 节点 [1,2,4]，token = [1,2,0]（节点 3 与 4 同被接受，取最深者）
    assert torch.equal(nodes, torch.tensor([1, 2, 4])), nodes
    # v*=4 是叶子 -> bonus 从 T[4]=[0.5,0.2,0.3]，CDF=[0.5,0.7,1.0]，u=0.25 -> 下标 0
    assert torch.equal(out, torch.tensor([1, 2, 0, 0])), out
    _run(solve, parent, token, q, T, u, 0.25)


def check_deepest_and_lexicographic_tiebreak(solve):
    # 两条同深存活路径 [2,1]（节点 3）与 [2,0]（节点 4）：按 token 字典序取更小的 [2,0]
    parent = torch.tensor([-1, 0, 0, 1, 2])
    token = torch.tensor([0, 2, 2, 1, 0])
    uni = [[1.0 / 3, 1.0 / 3, 1.0 / 3] for _ in range(5)]
    T = torch.tensor(uni)
    q = torch.tensor(uni)
    u = torch.tensor([0.0, 0.5, 0.5, 0.9, 0.9])
    out, nodes = solve(parent, token, q, T, u, 0.999)
    assert torch.equal(nodes, torch.tensor([2, 4])), nodes
    assert torch.equal(out, torch.tensor([2, 0, 2])), out
    _run(solve, parent, token, q, T, u, 0.999)

    # 把节点 4 改成必拒（p=0.05 vs q=0.8），最深存活点变成节点 3，路径 [2,1]，v*=3 是叶子走 bonus
    T2 = torch.tensor([[1.0 / 3, 1.0 / 3, 1.0 / 3],
                       [1.0 / 3, 1.0 / 3, 1.0 / 3],
                       [0.05, 0.5, 0.45],
                       [0.2, 0.3, 0.5],
                       [1.0 / 3, 1.0 / 3, 1.0 / 3]])
    q2 = torch.tensor([[1.0 / 3, 1.0 / 3, 1.0 / 3],
                       [1.0 / 3, 1.0 / 3, 1.0 / 3],
                       [1.0 / 3, 1.0 / 3, 1.0 / 3],
                       [1.0 / 3, 1.0 / 3, 1.0 / 3],
                       [0.8, 0.1, 0.1]])
    out, nodes = solve(parent, token, q2, T2, u, 0.6)
    assert torch.equal(nodes, torch.tensor([1, 3])), nodes
    assert torch.equal(out, torch.tensor([2, 1, 2])), out
    _run(solve, parent, token, q2, T2, u, 0.6)


def check_star_tree_one_level(solve):
    # 星型树（所有草稿都是根的孩子）：存活路径最多 1 个 token，收尾必走残差或 bonus
    rng = random.Random(99)
    for _ in range(30):
        M = rng.randint(2, 6)
        V = 3
        parent = [-1] + [0] * (M - 1)
        token = [0] + [rng.randrange(V) for _ in range(M - 1)]
        P, Q = [], []
        for _ in range(M):
            rp = [rng.random() + 0.01 for _ in range(V)]
            rq = [rng.random() + 0.01 for _ in range(V)]
            sp, sq = sum(rp), sum(rq)
            P.append([x / sp for x in rp])
            Q.append([x / sq for x in rq])
        u = [0.0] + [rng.random() for _ in range(M - 1)]
        uf = rng.random()
        pt, tt = torch.tensor(parent, dtype=torch.long), torch.tensor(token, dtype=torch.long)
        out, nodes = solve(pt, tt, torch.tensor(Q), torch.tensor(P), torch.tensor(u), uf)
        assert nodes.numel() <= 1, nodes
        assert out.numel() == nodes.numel() + 1, out
        _run(solve, pt, tt, torch.tensor(Q), torch.tensor(P), torch.tensor(u), uf)


check_example.example = True
