import torch

# 示例（由参考实现实跑回填）：1-D 五点、M=1（M0=2）、ef_construction=2
EXAMPLE_POINTS = [[0.0], [0.1], [0.05], [0.02], [4.0]]
EXAMPLE_LEVELS = [1, 0, 0, 0, 2]
EXAMPLE_M = 1
EXAMPLE_EF = 2
EXAMPLE_ADJ = [
    [[2, 3], [0], [0, 4], [0], [2]],
    [[4], [], [], [], [0]],
    [[], [], [], [], []],
]


def _sq(a, b):
    s = 0.0
    for u, v in zip(a, b):
        d = u - v
        s += d * d
    return s


def _oracle_hnsw(points, levels, M, ef_construction):
    """纯 Python oracle：列表排序代替堆，逐对显式累加距离，结构与参考解不同。"""
    pts = [[float(v) for v in row] for row in points.tolist()]
    n = len(pts)
    dmat = [[_sq(pts[i], pts[j]) for j in range(n)] for i in range(n)]
    lv = [int(v) for v in levels.tolist()]
    layer_count = max(lv) + 1
    max_level = lv[0]
    entry = 0
    adj = [[[] for _ in range(n)] for _ in range(layer_count)]

    def search_layer(q, ep, ef, lc):
        visited = [False] * n
        visited[ep] = True
        cand = [ep]
        best = [ep]
        while cand:
            cand.sort(key=lambda x: (dmat[q][x], x))
            c = cand.pop(0)
            best.sort(key=lambda x: (dmat[q][x], x))
            if dmat[q][c] > dmat[q][best[-1]]:
                break
            for e in sorted(adj[lc][c]):
                if visited[e]:
                    continue
                visited[e] = True
                de = dmat[q][e]
                if len(best) < ef or (de, e) < (dmat[q][best[-1]], best[-1]):
                    cand.append(e)
                    best.append(e)
                    best.sort(key=lambda x: (dmat[q][x], x))
                    if len(best) > ef:
                        best.pop()
        best.sort(key=lambda x: (dmat[q][x], x))
        return [(dmat[q][x], x) for x in best]

    def select_neighbors(q, cand_nodes, m):
        chosen = []
        for c in sorted(cand_nodes, key=lambda x: (dmat[q][x], x)):
            if len(chosen) >= m:
                break
            dcq = dmat[q][c]
            if all(dmat[c][r] >= dcq for r in chosen):
                chosen.append(c)
        return chosen

    for q in range(1, n):
        l = lv[q]
        ep = entry
        for lc in range(max_level, l, -1):
            found = search_layer(q, ep, 1, lc)
            ep = min(found, key=lambda t: (t[0], t[1]))[1]
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
                    far = max(adj[lc][e], key=lambda r: (dmat[e][r], r))
                    adj[lc][e].remove(far)
            adj[lc][q].sort()
            ep = min(found, key=lambda t: (t[0], t[1]))[1]
        if l > max_level:
            max_level = l
            entry = q

    return [[sorted(adj[lc][i]) for i in range(n)] for lc in range(layer_count)]


def _assert_valid(out, points, levels, M):
    n = points.shape[0]
    lv = levels.tolist()
    assert isinstance(out, list)
    assert len(out) == max(lv) + 1, (len(out), max(lv) + 1)
    for lc, layer in enumerate(out):
        assert isinstance(layer, list) and len(layer) == n
        for i, neigh in enumerate(layer):
            assert isinstance(neigh, list), type(neigh)
            assert all(isinstance(v, int) for v in neigh), neigh
            assert neigh == sorted(neigh), (lc, i, neigh)
            assert len(neigh) == len(set(neigh)), (lc, i, neigh)
            assert i not in neigh, (lc, i, neigh)
            assert len(neigh) <= (2 * M if lc == 0 else M), (lc, i, len(neigh))
            if lv[i] < lc:
                assert neigh == [], (lc, i, lv[i], neigh)
            for j in neigh:
                assert lv[j] >= lc, (lc, i, j, lv[j])
                assert 0 <= j < n


def _assert_same(got, exp):
    assert got == exp, (got, exp)


def check_example(solve):
    pts = torch.tensor(EXAMPLE_POINTS)
    lv = torch.tensor(EXAMPLE_LEVELS, dtype=torch.int64)
    out = solve(pts, lv, M=EXAMPLE_M, ef_construction=EXAMPLE_EF)
    _assert_same(out, EXAMPLE_ADJ)


check_example.example = True


def check_single_node(solve):
    pts = torch.tensor([[1.0, 2.0]])
    out = solve(pts, torch.tensor([2], dtype=torch.int64), M=2, ef_construction=4)
    assert out == [[[]], [[]], [[]]], out
    _assert_valid(out, pts, torch.tensor([2], dtype=torch.int64), 2)
    _assert_same(out, _oracle_hnsw(pts, torch.tensor([2], dtype=torch.int64), 2, 4))


def check_level_structure(solve):
    pts = torch.tensor([[0.0], [1.0], [2.0], [3.0], [10.0]])
    lv = torch.tensor([2, 0, 1, 0, 2], dtype=torch.int64)
    out = solve(pts, lv, M=2, ef_construction=4)
    _assert_valid(out, pts, lv, 2)
    _assert_same(out, _oracle_hnsw(pts, lv, 2, 4))
    assert out[1][1] == [] and out[1][3] == []
    assert out[2][1] == [] and out[2][2] == [] and out[2][3] == []
    assert out[2][0] or out[2][4]


def check_degree_caps_and_overflow(solve):
    """M=1、M0=2：节点 0 的层 0 出度一度到 3，必须按最远邻居剔除。"""
    pts = torch.zeros(4, 1)
    lv = torch.tensor([1, 0, 0, 0], dtype=torch.int64)
    out = solve(pts, lv, M=1, ef_construction=2)
    exp = [[[1, 2], [0], [0], [0]], [[], [], [], []]]
    _assert_same(out, exp)
    # 溢出时并列距离取节点号大者剔除：0 的邻居 1,2,3 距离全 0，删掉 3。
    assert 3 not in out[0][0]
    assert out[0][3] == [0]          # 反向边保留，最终图可以是不对称的
    _assert_valid(out, pts, lv, 1)
    _assert_same(out, _oracle_hnsw(pts, lv, 1, 2))


def check_duplicate_points_deterministic(solve):
    pts = torch.zeros(6, 2)
    lv = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.int64)
    a = solve(pts, lv, M=2, ef_construction=3)
    b = solve(pts, lv, M=2, ef_construction=3)
    _assert_same(a, b)
    _assert_valid(a, pts, lv, 2)
    _assert_same(a, _oracle_hnsw(pts, lv, 2, 3))


def check_oracle_cases(solve):
    cases = [
        (torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]]),
         torch.tensor([0, 0, 0, 0, 0], dtype=torch.int64), 2, 4),
        (torch.tensor([[0.0], [0.5], [1.0], [1.5]]),
         torch.tensor([3, 2, 1, 0], dtype=torch.int64), 2, 3),
        (torch.tensor([[0.0], [0.5], [1.0], [1.5]]),
         torch.tensor([0, 1, 2, 3], dtype=torch.int64), 1, 2),
        (torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0],
                       [5.0, 5.0], [5.0, 6.0]]),
         torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.int64), 2, 4),
        (torch.tensor([[0.25, -1.0], [0.25, 1.0], [3.0, 0.5], [3.0, 0.75]],
                      dtype=torch.float64),
         torch.tensor([0, 0, 1, 0], dtype=torch.int64), 2, 3),
    ]
    for pts, lv, m, ef in cases:
        out = solve(pts, lv, M=m, ef_construction=ef)
        _assert_valid(out, pts, lv, m)
        _assert_same(out, _oracle_hnsw(pts, lv, m, ef))


def check_ef_smaller_than_m(solve):
    pts = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    lv = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
    out = solve(pts, lv, M=3, ef_construction=1)
    _assert_valid(out, pts, lv, 3)
    _assert_same(out, _oracle_hnsw(pts, lv, 3, 1))


def check_errors(solve):
    pts = torch.tensor(EXAMPLE_POINTS)
    lv = torch.tensor(EXAMPLE_LEVELS, dtype=torch.int64)

    def bad(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    bad(lambda: solve(torch.zeros(0, 1), torch.zeros(0, dtype=torch.int64)))
    bad(lambda: solve(torch.zeros(2, 0), torch.zeros(2, dtype=torch.int64)))
    bad(lambda: solve(torch.zeros(2, 1, 1), torch.zeros(2, dtype=torch.int64)))
    bad(lambda: solve(torch.zeros(2, 1, dtype=torch.int64),
                      torch.zeros(2, dtype=torch.int64)))
    bad(lambda: solve(pts, lv.to(torch.int32)))
    bad(lambda: solve(pts, lv[:2]))
    bad(lambda: solve(pts, lv.reshape(5, 1)))
    bad(lambda: solve(pts, torch.tensor([-1, 0, 0, 0, 0], dtype=torch.int64)))
    bad(lambda: solve(pts, lv, M=0))
    bad(lambda: solve(pts, lv, M=1.5))
    bad(lambda: solve(pts, lv, M=True))
    bad(lambda: solve(pts, lv, ef_construction=0))
