"""参考答案：EAGLE-2 动态草稿树重排与剪枝（arXiv:2406.16858，Algorithm 1 的逐层贪心实现）。"""

import torch


def solve(parent, acc, k, gmax):
    """EAGLE-2 动态草稿树：先按静态树的父子结构算累积分数 V(v)=∏ acc（自根到 v 的链上），
    再逐层（breadth）从当前保留节点展开候选，按 (-V, index) 排序取「剩余预算」个，直到预算
    用满 / 无候选 / 触到层数上限 gmax。返回 (选中节点索引升序 long 张量, 期望吐词 1+ΣV)。"""
    M = int(parent.shape[0])
    if M <= 1:
        return torch.zeros(0, dtype=torch.long), 1.0
    kids = [[] for _ in range(M)]
    for v in range(1, M):
        kids[int(parent[v])].append(v)
    V = [1.0] * M
    for v in range(1, M):
        V[v] = V[int(parent[v])] * float(acc[v])

    frontier = [0]
    chosen = []
    used = 0
    level = 0
    while frontier and used < k and level < gmax:
        cands = [c for u in frontier for c in kids[u]]
        if not cands:
            break
        cands.sort(key=lambda v: (-V[v], v))
        take = cands[: k - used]
        chosen.extend(take)
        used += len(take)
        frontier = take
        level += 1

    sel = sorted(chosen)
    E = 1.0 + sum(V[v] for v in sel)
    return torch.tensor(sel, dtype=torch.long), float(E)
