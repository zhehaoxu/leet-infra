def solve(n, edges, cost):
    """把子任务依赖图排成执行计划。

    n:任务数，编号 0..n-1；edges:[(u, v)] 表示 u 必须先于 v；cost:每个任务的耗时（> 0）。

    1) 拓扑序：Kahn 算法，就绪集合里**总取编号最小**的那个（保证结果唯一）；
    2) 并行波次 waves：level[v] = 0（无前驱）否则 1 + max(level[u] for u->v)，
       第 k 波是所有 level == k 的任务，波内按编号升序；
    3) 加权关键路径 crit：任意一条依赖链上 cost 之和的最大值（决定最短总工期）。

    有环、编号越界、cost 长度不符或非正 → ValueError。
    返回 (order, waves, crit)，crit 为 float。
    """
    # TODO
    raise NotImplementedError
