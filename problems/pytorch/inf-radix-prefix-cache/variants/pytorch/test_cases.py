"""独立 oracle：用「完整路径元组 → (最近使用时刻, 编号)」的扁平表模拟缓存的另一套写法。

参考实现用带父指针的压缩边节点树，本 oracle 只存每条路径的完整 token 元组，父子关系由
最长真前缀隐式恢复，两边不共享任何数据结构。
"""

import random


def _oracle(capacity, ops):
    nodes = {(): (-1, 0)}
    next_id = [0]
    active = {}

    def parent_of(path):
        for k in range(len(path) - 1, -1, -1):
            if path[:k] in nodes:
                return path[:k]
        return ()

    def total():
        return sum(len(p) - len(parent_of(p)) for p in nodes if p)

    def pinned(path):
        size = len(path)
        return any(len(q) >= size and q[:size] == path for q in active.values())

    def evict():
        freed = 0
        while total() > capacity:
            leaves = [
                p
                for p in nodes
                if p and not any(q != p and q[: len(p)] == p for q in nodes)
            ]
            free_leaves = [p for p in leaves if not pinned(p)]
            if not free_leaves:
                break
            victim = min(free_leaves, key=lambda p: (nodes[p][0], nodes[p][1]))
            freed += len(victim) - len(parent_of(victim))
            del nodes[victim]
        return freed

    hits = []
    evicted = 0
    for now, op in enumerate(ops):
        if op[0] == "finish":
            active.pop(op[1], None)
            evicted += evict()
            continue

        _, rid, tokens = op
        t = tuple(tokens)
        best = ()
        for p in nodes:
            if len(p) <= len(t) and t[: len(p)] == p and len(p) > len(best):
                best = p
        for p in list(nodes):
            if len(p) <= len(best) and best[: len(p)] == p:
                nodes[p] = (now, nodes[p][1])

        pos = len(best)
        if pos < len(t):
            child = None
            for p in nodes:
                if (
                    len(p) > len(best)
                    and p[: len(best)] == best
                    and parent_of(p) == best
                    and p[pos] == t[pos]
                ):
                    child = p
                    break
            if child is not None:
                k = 0
                while (
                    k < len(child) - pos
                    and pos + k < len(t)
                    and child[pos + k] == t[pos + k]
                ):
                    k += 1
                split = pos + k
                if split < len(child):
                    next_id[0] += 1
                    nodes[t[:split]] = (now, next_id[0])
                pos = split
            if pos < len(t):
                next_id[0] += 1
                nodes[t] = (now, next_id[0])

        hits.append(pos)
        active[rid] = t
        evicted += evict()

    return hits, evicted, total()


def check_example(solve):
    """最长公共前缀 + 边中途劈分：后来的请求只命中已有的那段。"""
    hits, evicted, cached = solve(
        32,
        [
            ("request", 0, [10, 11, 12, 13]),
            ("finish", 0),
            ("request", 1, [10, 11, 99]),
            ("finish", 1),
            ("request", 2, [10, 11, 12]),
            ("finish", 2),
            ("request", 3, [10]),
            ("finish", 3),
        ],
    )
    assert hits == [0, 2, 3, 1], f"命中数应为 [0, 2, 3, 1]，got {hits}"
    assert evicted == 0, f"容量充足时不应驱逐，got {evicted}"
    assert cached == 5, f"最终缓存 5 个 token，got {cached}"


check_example.example = True


def check_lru_victim(solve):
    """超出容量时驱逐最近使用时刻最小的叶子，而不是最早创建的节点。"""
    hits, evicted, cached = solve(
        6,
        [
            ("request", 0, [1, 2, 3]),
            ("finish", 0),
            ("request", 1, [4, 5, 6]),
            ("finish", 1),
            ("request", 2, [7, 8, 9]),
            ("finish", 2),
            ("request", 3, [4, 5, 6]),
            ("finish", 3),
        ],
    )
    assert hits == [0, 0, 0, 3], f"[1,2,3] 应被静默驱逐，got {hits}"
    assert evicted == 3, f"got {evicted}"
    assert cached == 6, f"got {cached}"


def check_hit_refreshes_lru(solve):
    """命中会刷新最近使用时刻，于是被驱逐的是更久没被碰过的另一条分支。"""
    hits, evicted, cached = solve(
        6,
        [
            ("request", 0, [1, 2, 3]),
            ("finish", 0),
            ("request", 1, [4, 5, 6]),
            ("finish", 1),
            ("request", 2, [1, 2, 3]),
            ("finish", 2),
            ("request", 3, [7, 8, 9]),
            ("finish", 3),
            ("request", 4, [1, 2, 3]),
            ("finish", 4),
        ],
    )
    assert hits == [0, 0, 3, 0, 3], f"[1,2,3] 刚被命中过，应留下来，got {hits}"
    assert evicted == 3, f"只该驱逐 [4,5,6]，got {evicted}"
    assert cached == 6, f"got {cached}"


def check_pinned_exceeds_capacity(solve):
    """活跃请求占住整条路径，驱逐找不到可动的叶子，缓存允许暂时超限。"""
    hits, evicted, cached = solve(
        4,
        [
            ("request", 0, [1, 2, 3, 4]),
            ("request", 1, [5, 6]),
            ("finish", 0),
        ],
    )
    assert hits == [0, 0], f"got {hits}"
    assert evicted == 4, f"finish 释放后 [1,2,3,4] 才可驱逐，got {evicted}"
    assert cached == 2, f"只剩 [5,6]，got {cached}"


def check_parent_kept_after_leaf_eviction(solve):
    """驱逐只摘叶子边：父路径留在缓存里，后来者仍能命中它。"""
    hits, evicted, cached = solve(
        5,
        [
            ("request", 0, [1, 2]),
            ("finish", 0),
            ("request", 1, [1, 2, 3, 4]),
            ("finish", 1),
            ("request", 2, [7, 7, 7]),
            ("finish", 2),
            ("request", 3, [7]),
            ("finish", 3),
        ],
    )
    assert hits == [0, 2, 0, 1], f"got {hits}"
    assert evicted == 2, f"被摘掉的是 [3,4] 这条边，got {evicted}"
    assert cached == 5, f"got {cached}"


def check_no_shared_prefix(solve):
    """完全不同的 prompt 之间没有任何命中，缓存只是逐条堆积。"""
    hits, evicted, cached = solve(
        64,
        [
            ("request", 0, [1]),
            ("finish", 0),
            ("request", 1, [2]),
            ("finish", 1),
            ("request", 2, [3, 4]),
            ("finish", 2),
        ],
    )
    assert hits == [0, 0, 0], f"got {hits}"
    assert (evicted, cached) == (0, 4), (evicted, cached)


def check_request_not_finished_is_not_reusable_slot(solve):
    """未 finish 的请求把路径钉住；此时更老的未钉住叶子反而先被驱逐。"""
    hits, evicted, cached = solve(
        5,
        [
            ("request", 0, [1, 2]),
            ("finish", 0),
            ("request", 1, [3, 4, 5]),
            ("request", 2, [9]),
        ],
    )
    assert hits == [0, 0, 0], f"got {hits}"
    assert evicted == 2, f"r1 还在跑，只能动更老的 [1,2]，got {evicted}"
    assert cached == 4, f"got {cached}"


def check_random_against_oracle(solve):
    """随机请求流与独立 oracle 全等，并核对账本不变量。"""
    rng = random.Random(487)
    for _ in range(300):
        capacity = rng.randint(1, 12)
        n = rng.randint(1, 6)
        ops = []
        live = []
        for rid in range(n):
            length = rng.randint(1, 6)
            tokens = [rng.randint(0, 3) for _ in range(length)]
            ops.append(("request", rid, tokens))
            live.append(rid)
            for done in list(live):
                if rng.random() < 0.35:
                    ops.append(("finish", done))
                    live.remove(done)
        for done in live:
            ops.append(("finish", done))

        got = solve(capacity, ops)
        want = _oracle(capacity, ops)
        assert got == want, f"{capacity} {ops}: {got} vs {want}"

        hits, evicted, cached = got
        assert len(hits) == n, f"每个 request 恰好一个命中数，got {len(hits)}"
        prompts = [o[2] for o in ops if o[0] == "request"]
        assert all(0 <= h <= len(p) for h, p in zip(hits, prompts)), (hits, prompts)
        inserted = sum(len(p) for p in prompts) - sum(hits)
        assert cached == inserted - evicted, (
            f"缓存账不平：{inserted} 插入 - {evicted} 驱逐 != {cached}"
        )
