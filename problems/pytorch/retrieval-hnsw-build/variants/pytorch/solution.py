"""参考答案：HNSW 建图插入与 Alg.4 启发式邻居选择（确定性，无随机数）。

层级由 levels 入参显式给出；按插入序 0..N-1 逐点插入。上层用 ef=1 的
SEARCHLAYER 贪心下降，本层及以下用 ef_construction 宽度的候选搜索；候选
按 (平方欧氏距离, 节点号) 排序后做 Alg.4 启发式（若候选到某个已选邻居的
距离严格小于它到新点的距离则剪掉），选满 M 个后双向连边；已有节点出度
超过 cap（第 0 层 2M，其余层 M）时反复删掉距它最远的邻居（并列删节点号
大者）。返回 L=max(levels)+1 层、每层 N 个升序邻居列表。
"""

import heapq

import torch


def solve(points, levels, M=16, ef_construction=200):
    if not isinstance(points, torch.Tensor):
        raise ValueError("points 必须是 torch.Tensor")
    if points.dim() != 2:
        raise ValueError("points 必须是二维张量")
    if points.dtype not in (torch.float32, torch.float64):
        raise ValueError("points 的 dtype 必须是 float32 或 float64")
    n, d = points.shape
    if n == 0:
        raise ValueError("points 至少需要一个点")
    if d == 0:
        raise ValueError("points 的特征维至少为 1")
    if not isinstance(levels, torch.Tensor):
        raise ValueError("levels 必须是 torch.Tensor")
    if levels.dtype != torch.int64:
        raise ValueError("levels 的 dtype 必须是 int64")
    if levels.dim() != 1 or levels.numel() != n:
        raise ValueError("levels 必须是一维且长度等于 N")
    if bool((levels < 0).any()):
        raise ValueError("levels 必须非负")
    if isinstance(M, bool) or not isinstance(M, int) or M < 1:
        raise ValueError("M 必须是正整数")
    if isinstance(ef_construction, bool) or not isinstance(ef_construction, int) \
            or ef_construction < 1:
        raise ValueError("ef_construction 必须是正整数")

    x = points.to(torch.float64)
    dist = ((x[:, None, :] - x[None, :, :]) ** 2).sum(dim=2).tolist()
    lv = levels.tolist()
    layer_count = max(lv) + 1
    max_level = lv[0]
    entry = 0
    adj = [[[] for _ in range(n)] for _ in range(layer_count)]

    def dl(a, b):
        return dist[a][b]

    def search_layer(q, ep, ef, lc):
        visited = {ep}
        cand = [(dl(q, ep), ep)]
        best = [(-dl(q, ep), -ep)]
        while cand:
            dc, c = heapq.heappop(cand)
            worst_d = -best[0][0]
            if dc > worst_d:
                break
            for e in sorted(adj[lc][c]):
                if e in visited:
                    continue
                visited.add(e)
                de = dl(q, e)
                worst_d, worst_n = -best[0][0], -best[0][1]
                if len(best) < ef or (de, e) < (worst_d, worst_n):
                    heapq.heappush(cand, (de, e))
                    heapq.heappush(best, (-de, -e))
                    if len(best) > ef:
                        heapq.heappop(best)
        return [(-nd, -nn) for nd, nn in best]

    def select_neighbors(q, cand_nodes, m):
        ordered = sorted(cand_nodes, key=lambda c: (dl(q, c), c))
        chosen = []
        for c in ordered:
            if len(chosen) >= m:
                break
            dcq = dl(q, c)
            if all(dl(c, r) >= dcq for r in chosen):
                chosen.append(c)
        return chosen

    for q in range(1, n):
        l = lv[q]
        ep = entry
        for lc in range(max_level, l, -1):
            found = search_layer(q, ep, 1, lc)
            ep = min(found, key=lambda t: (dl(q, t[1]), t[1]))[1]
        for lc in range(min(max_level, l), -1, -1):
            found = search_layer(q, ep, ef_construction, lc)
            neighbors = select_neighbors(q, [node for _, node in found], M)
            for e in neighbors:
                if e not in adj[lc][q]:
                    adj[lc][q].append(e)
                if q not in adj[lc][e]:
                    adj[lc][e].append(q)
            cap = 2 * M if lc == 0 else M
            for e in neighbors:
                while len(adj[lc][e]) > cap:
                    far = max(adj[lc][e], key=lambda r: (dl(e, r), r))
                    adj[lc][e].remove(far)
            adj[lc][q].sort()
            ep = min(found, key=lambda t: (dl(q, t[1]), t[1]))[1]
        if l > max_level:
            max_level = l
            entry = q

    return [[sorted(adj[lc][i]) for i in range(n)] for lc in range(layer_count)]
