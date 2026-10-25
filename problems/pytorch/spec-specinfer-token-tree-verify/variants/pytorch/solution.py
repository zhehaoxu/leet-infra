"""参考答案：SpecInfer 通用 token-tree 并行验证（arXiv:2305.09781）。"""

import torch


def solve(parent, token, qdist, T, u, u_final):
    """沿接受前缀逐层推进地验证整棵草稿树；某层全军覆没即停止，取最深存活路径，
    路径末端若挂着被拒孩子就用 residual=norm(max(T[v*]-qdist[c],0)) 出收尾 token，
    否则用 T[v*] 出 bonus token。"""
    M = int(parent.shape[0])
    Pl = parent.tolist()
    Tk = token.tolist()
    Ul = u.tolist()
    Tl = T.tolist()
    Ql = qdist.tolist()

    def cdf(row, uu):
        c = 0.0
        for i, x in enumerate(row):
            c += x
            if c > uu:
                return i
        return len(row) - 1

    kids = [[] for _ in range(M)]
    for v in range(1, M):
        kids[Pl[v]].append(v)

    accepted = {0}
    frontier = [0]
    while frontier:
        cands = sorted(c for node in frontier for c in kids[node])
        if not cands:
            break
        kept = []
        for v in cands:
            q = Ql[v][Tk[v]]
            p = Tl[Pl[v]][Tk[v]]
            ratio = p / q if q > 0 else float("inf")
            if Ul[v] < min(1.0, ratio):
                kept.append(v)
        if not kept:
            break
        accepted.update(kept)
        frontier = kept

    best_key, last = None, 0
    for v in sorted(accepted):
        toks, cur = [], v
        while cur != 0:
            toks.append(Tk[cur])
            cur = Pl[cur]
        toks.reverse()
        key = (-len(toks), toks, v)
        if best_key is None or key < best_key:
            best_key, last = key, v

    nodes = []
    cur = last
    while cur != 0:
        nodes.append(cur)
        cur = Pl[cur]
    nodes.reverse()
    path = [Tk[v] for v in nodes]

    rejected_kids = [c for c in kids[last] if c not in accepted]
    if rejected_kids:
        c0 = min(rejected_kids)
        row = [max(Tl[last][a] - Ql[c0][a], 0.0) for a in range(len(Tl[last]))]
        s = sum(row)
        row = [x / s for x in row]
    else:
        row = Tl[last]
    path.append(cdf(row, float(u_final)))
    return torch.tensor(path, dtype=torch.long), torch.tensor(nodes, dtype=torch.long)
