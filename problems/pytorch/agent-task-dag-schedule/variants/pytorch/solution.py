import heapq


def _check(n, edges, cost):
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n 必须是 >= 1 的整数")
    if len(cost) != n:
        raise ValueError(f"cost 长度应为 {n}，got {len(cost)}")
    if any(c <= 0 for c in cost):
        raise ValueError("cost 必须全为正")
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("edges 的每一项必须是 (u, v)")
        u, v = edge
        if not (0 <= u < n) or not (0 <= v < n):
            raise ValueError(f"边 ({u}, {v}) 的端点越界")


def solve(n, edges, cost):
    _check(n, edges, cost)

    preds = set()
    succs = [[] for _ in range(n)]
    for u, v in edges:
        if u == v:
            raise ValueError(f"自环 ({u}, {u}) 构成环")
        if (u, v) in preds:
            continue  # 去重边
        preds.add((u, v))
        succs[u].append(v)

    indeg = [0] * n
    for u in range(n):
        for v in succs[u]:
            indeg[v] += 1

    best = [0.0] * n  # 所有前驱中最长链的 cost 之和
    level = [0] * n
    ready = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(ready)
    order = []

    while ready:
        u = heapq.heappop(ready)
        order.append(u)
        reach = best[u] + float(cost[u])
        for v in succs[u]:
            if reach > best[v]:
                best[v] = reach
            if level[u] + 1 > level[v]:
                level[v] = level[u] + 1
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(ready, v)

    if len(order) != n:
        raise ValueError("依赖图有环，无法拓扑排序")

    depth = max(level) + 1
    waves = [[] for _ in range(depth)]
    for v in range(n):
        waves[level[v]].append(v)  # 按编号升序遍历，波内天然升序

    crit = max(best[v] + float(cost[v]) for v in range(n))
    return order, waves, crit
