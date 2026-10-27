"""独立 oracle：暴力选最小就绪点 / 松弛法求 level / 记忆化 DFS 求关键路径。"""


def _preds(n, edges):
    preds = [[] for _ in range(n)]
    seen = set()
    for u, v in edges:
        if (u, v) in seen:
            continue
        seen.add((u, v))
        preds[v].append(u)
    return preds


def _oracle_order(n, preds):
    done, order = set(), []
    while len(order) < n:
        pick = None
        for v in range(n):
            if v in done:
                continue
            if all(u in done for u in preds[v]):
                pick = v
                break
        if pick is None:
            raise ValueError("cycle")
        done.add(pick)
        order.append(pick)
    return order


def _oracle_levels(n, preds):
    level = [0] * n
    changed = True
    while changed:
        changed = False
        for v in range(n):
            for u in preds[v]:
                if level[u] + 1 > level[v]:
                    level[v] = level[u] + 1
                    changed = True
    return level


def _oracle_crit(n, preds, cost):
    memo = {}

    def best(v):
        if v not in memo:
            base = 0.0
            for u in preds[v]:
                base = max(base, best(u))
            memo[v] = base + float(cost[v])
        return memo[v]

    return max(best(v) for v in range(n))


_N, _EDGES, _COST = 6, [(0, 2), (1, 2), (2, 3), (2, 4), (4, 5)], [3, 2, 4, 1, 5, 2]


def check_example(solve):
    order, waves, crit = solve(_N, _EDGES, _COST)
    assert order == [0, 1, 2, 3, 4, 5], f"拓扑序不对：{order}"
    assert waves == [[0, 1], [2], [3, 4], [5]], f"波次不对：{waves}"
    assert abs(crit - 14.0) < 1e-9, f"关键路径应为 14，got {crit}"


check_example.example = True


def check_order_tiebreak(solve):
    # 0,2,3 三路并行；1 依赖 3，所以必须排在 3 后面（不是简单的编号顺序）
    order, _, _ = solve(4, [(3, 1)], [1, 1, 1, 1])
    assert order == [0, 2, 3, 1], f"应总取编号最小的就绪点：{order}"

    # 与暴力 oracle 对拍：随机 DAG（(i*7+j*3)%5 过滤，确定性）
    n = 12
    edges = [(i, j) for i in range(n) for j in range(i + 2, n) if (i * 7 + j * 3) % 5 == 0]
    cost = [1 + (i * 3) % 4 for i in range(n)]
    got = solve(n, edges, cost)
    oracle = _oracle_order(n, _preds(n, edges))
    assert got[0] == oracle, f"拓扑序与 oracle 不一致：{got[0]} vs {oracle}"


def check_waves(solve):
    n = 7
    edges = [(0, 3), (1, 3), (2, 4), (3, 5), (4, 5), (5, 6)]
    cost = [2, 1, 3, 1, 2, 1, 4]
    order, waves, _ = solve(n, edges, cost)

    preds = _preds(n, edges)
    flat = [v for wave in waves for v in wave]
    assert sorted(flat) == list(range(n)), f"波次应恰好覆盖所有任务：{flat}"
    for wave in waves:
        assert wave == sorted(wave), f"波内应按编号升序：{waves}"
    assert waves == [[0, 1, 2], [3, 4], [5], [6]], waves

    level = _oracle_levels(n, preds)
    assert [level[v] for v in flat] == [i for i, wave in enumerate(waves) for _ in wave]
    assert max(len(w) for w in waves) == 3, "最大并行度应为 3"
    assert order == _oracle_order(n, preds), order


def check_crit(solve):
    # 菱形：两条支路代价不同，关键路径取长的那条（加权而非跳数）
    n = 4
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    cost = [1, 5, 2, 1]
    _, _, crit = solve(n, edges, cost)
    assert abs(crit - 7.0) < 1e-9, f"应走 0->1->3 = 7，got {crit}"

    # 孤立点也参与关键路径（等于自身耗时）
    _, _, crit = solve(3, [], [4, 1, 9])
    assert abs(crit - 9.0) < 1e-9, crit

    # 与记忆化 DFS oracle 对拍
    n = 12
    edges = [(i, j) for i in range(n) for j in range(i + 2, n) if (i * 5 + j) % 4 == 0]
    cost = [1 + (i * 7) % 5 for i in range(n)]
    got = solve(n, edges, cost)
    assert abs(got[2] - _oracle_crit(n, _preds(n, edges), cost)) < 1e-9, got[2]
    assert got[0] == _oracle_order(n, _preds(n, edges))


def check_cycle_and_invalid(solve):
    for bad in (
        lambda: solve(2, [(0, 1), (1, 0)], [1, 1]),
        lambda: solve(1, [(0, 0)], [1]),  # 自环
        lambda: solve(3, [(0, 1), (1, 2), (2, 0)], [1, 1, 1]),
        lambda: solve(2, [(0, 3)], [1, 1]),  # 端点越界
        lambda: solve(2, [(0, 1)], [1]),  # cost 长度不符
        lambda: solve(2, [(0, 1)], [1, 0]),  # 非正耗时
        lambda: solve(0, [], []),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass

    # 重复边不去重也不应该有副作用
    assert solve(2, [(0, 1), (0, 1)], [1, 2]) == ([0, 1], [[0], [1]], 3.0)
