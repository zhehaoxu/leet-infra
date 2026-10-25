import itertools


# 独立重算参考：暴力枚举全部保留子集（结构与参考解的背包 DP 完全不同），
# 过滤满足「Σmem≤budget 且 |K|≥min_keep」的集合，取 Σcost(补集) 最小值。
def _oracle(mem, cost, budget, min_keep):
    mem = [int(x) for x in mem]
    cost = [int(x) for x in cost]
    N = len(mem)
    total = sum(cost)
    best = None
    for r in range(min_keep, N + 1):
        for K in itertools.combinations(range(N), r):
            if sum(mem[i] for i in K) <= budget:
                rc = total - sum(cost[i] for i in K)
                if best is None or rc < best:
                    best = rc
    if best is None:
        raise ValueError("infeasible")
    return best


def _call(fn, mem, cost, B, L):
    try:
        return ("ok", fn(mem, cost, B, L))
    except ValueError:
        return ("infeasible", None)


def check_example(solve):
    assert solve([30, 10, 20, 40, 15], [50, 20, 30, 60, 10], 60, 2) == 70


def check_matches_oracle(solve):
    configs = [
        ([30, 10, 20, 40, 15], [50, 20, 30, 60, 10], 60, 2),
        ([30, 10, 20, 40, 15], [50, 20, 30, 60, 10], 100, 1),
        ([30, 10, 20, 40, 15], [50, 20, 30, 60, 10], 45, 2),
        ([30, 10, 20, 40, 15], [50, 20, 30, 60, 10], 0, 0),
        ([30, 10, 20, 40, 15], [50, 20, 30, 60, 10], 115, 0),
        ([5, 5, 5], [1, 2, 3], 10, 2),
        ([8, 4, 6, 2], [10, 4, 7, 3], 10, 2),
        ([12, 1, 1, 1, 1], [100, 5, 5, 5, 5], 4, 4),
        ([3, 7, 2, 9, 4, 1], [8, 4, 6, 2, 5, 9], 12, 3),
        # 故意放一个不可行用例：min_keep=3 三个 mem=5 需 15 > budget 10
        ([5, 5, 5], [1, 2, 3], 10, 3),
    ]
    for mem, cost, B, L in configs:
        assert _call(solve, mem, cost, B, L) == _call(_oracle, mem, cost, B, L), (mem, cost, B, L)


def check_keep_all_zero_recompute(solve):
    # 预算足够大时保留全部，重算代价为 0
    mem = [30, 10, 20, 40, 15]
    cost = [50, 20, 30, 60, 10]
    assert solve(mem, cost, sum(mem), 0) == 0
    assert solve(mem, cost, sum(mem), 5) == 0


def check_min_keep_binds(solve):
    # 高 cost 的 0 号 mem=100 独占预算：只留它 1 块 cost 最大，但 min_keep=4 逼着你改留 4 个便宜小块
    mem = [100, 1, 1, 1, 1]
    cost = [1000, 5, 5, 5, 5]
    # min_keep=1：预算 100 留 {0} cost=1000 → 重算 20（贪心只留贵的）
    assert solve(mem, cost, 100, 1) == 20
    # min_keep=4：凑够 4 块只能选 {1,2,3,4}（mem=4）cost=20 → 重算 1000（被保底数逼高）
    assert solve(mem, cost, 100, 4) == 1000


def check_infeasible_raises(solve):
    mem = [30, 10, 20, 40, 15]
    cost = [50, 20, 30, 60, 10]
    # 最小的 min_keep=2 个 mem 之和 = 10+15 = 25 > 20 -> 不可行
    try:
        solve(mem, cost, 20, 2)
        assert False, "expected ValueError"
    except ValueError:
        pass
    # min_keep 超过 N 也不可行
    try:
        solve(mem, cost, 1000, 6)
        assert False, "expected ValueError"
    except ValueError:
        pass


check_example.example = True
