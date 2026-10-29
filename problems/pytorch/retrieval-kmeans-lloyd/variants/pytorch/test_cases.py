import torch

# 示例期望值（由参考实现实跑回填）
EXAMPLE_POINTS = [[0.0, 0.0], [0.0, 2.0], [10.0, 0.0], [10.0, 2.0]]
EXAMPLE_INIT = [[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]]
EXAMPLE_LABELS = [0, 1, 2, 2]
EXAMPLE_CENTROIDS = [[0.0, 0.0], [0.0, 2.0], [10.0, 1.0]]
EXAMPLE_ITERS = 3


def _sq(a, b):
    s = 0.0
    for u, v in zip(a, b):
        d = u - v
        s += d * d
    return s


def _oracle_kmeans(points, init_centroids, tol=1e-4, max_iters=100):
    """纯 Python 逐点/逐轮重算：显式三重循环距离、严格 < 更新 argmin。"""
    pts = [[float(v) for v in row] for row in points.tolist()]
    cent = [[float(v) for v in row] for row in init_centroids.tolist()]
    n, k = len(pts), len(cent)
    iters = 0
    for _ in range(max_iters):
        iters += 1
        dist = [[_sq(pts[i], cent[j]) for j in range(k)] for i in range(n)]
        labels = []
        for i in range(n):
            best, best_d = 0, dist[i][0]
            for j in range(1, k):
                if dist[i][j] < best_d:
                    best, best_d = j, dist[i][j]
            labels.append(best)
        counts = [0] * k
        for t in labels:
            counts[t] += 1
        new_cent = [row[:] for row in cent]
        for j in range(k):
            if counts[j] > 0:
                new_cent[j] = [
                    sum(pts[i][t] for i in range(n) if labels[i] == j) / counts[j]
                    for t in range(len(pts[0]))
                ]
        empty = [j for j in range(k) if counts[j] == 0]
        if empty:
            nearest = [min(dist[i]) for i in range(n)]
            order = sorted(range(n), key=lambda i: (-nearest[i], i))
            used = set()
            for j in empty:
                for i in order:
                    if i not in used:
                        used.add(i)
                        new_cent[j] = pts[i][:]
                        break
        disp = 0.0
        for j in range(k):
            disp = max(disp, sum((new_cent[j][t] - cent[j][t]) ** 2
                                 for t in range(len(cent[0]))) ** 0.5)
        cent = new_cent
        if disp < tol:
            break
    # 最终质心 cast 回输入 dtype，再据此重算标签
    cent_cast = torch.tensor(cent, dtype=points.dtype).to(torch.float64).tolist()
    final_labels = []
    for i in range(n):
        best, best_d = 0, _sq(pts[i], cent_cast[0])
        for j in range(1, k):
            d = _sq(pts[i], cent_cast[j])
            if d < best_d:
                best, best_d = j, d
        final_labels.append(best)
    return (
        torch.tensor(final_labels, dtype=torch.int64),
        torch.tensor(cent_cast, dtype=points.dtype),
        iters,
    )


def _atol(dtype):
    return 1e-5 if dtype == torch.float32 else 1e-9


def _assert_matches(solve, points, init, tol=1e-4, max_iters=100):
    got_l, got_c, got_it = solve(points, init, tol=tol, max_iters=max_iters)
    exp_l, exp_c, exp_it = _oracle_kmeans(points, init, tol, max_iters)
    assert got_l.dtype == torch.int64, got_l.dtype
    assert got_l.shape == (points.shape[0],), got_l.shape
    assert got_c.shape == init.shape and got_c.dtype == points.dtype
    assert isinstance(got_it, int), type(got_it)
    assert torch.equal(got_l, exp_l), (got_l.tolist(), exp_l.tolist())
    assert got_it == exp_it, (got_it, exp_it)
    assert torch.allclose(got_c, exp_c, atol=_atol(points.dtype), rtol=0), \
        (got_c.tolist(), exp_c.tolist())


def check_example(solve):
    pts = torch.tensor(EXAMPLE_POINTS)
    init = torch.tensor(EXAMPLE_INIT)
    labels, cent, iters = solve(pts, init, tol=1e-4, max_iters=100)
    assert torch.equal(labels, torch.tensor(EXAMPLE_LABELS)), labels.tolist()
    assert labels.dtype == torch.int64 and labels.shape == (4,)
    assert torch.equal(cent, torch.tensor(EXAMPLE_CENTROIDS)), cent.tolist()
    assert cent.dtype == torch.float32 and cent.shape == (3, 2)
    assert iters == EXAMPLE_ITERS, iters


check_example.example = True


def check_empty_cluster_reseed(solve):
    """第 1 轮簇 1 为空；重播种把 p1=[0,2] 搬成新质心，之后标签才稳定。"""
    pts = torch.tensor(EXAMPLE_POINTS)
    init = torch.tensor(EXAMPLE_INIT)
    _, cent1, it1 = solve(pts, init, max_iters=1)
    # 只跑 1 轮：c0 已变成 [0,1]，c2=[10,1]，空簇 c1 被重播种为 [0,2]
    assert torch.equal(cent1, torch.tensor([[0.0, 1.0], [0.0, 2.0], [10.0, 1.0]])), \
        cent1.tolist()
    assert it1 == 1
    labels, cent, iters = solve(pts, init)
    assert torch.equal(cent, torch.tensor(EXAMPLE_CENTROIDS)), cent.tolist()
    assert torch.equal(labels, torch.tensor(EXAMPLE_LABELS)), labels.tolist()
    assert iters == 3
    # 重播种的簇 1 最终质心恰好等于某个输入点（搬点而非取均值）
    assert any(torch.equal(cent[1], p) for p in pts)


def check_convergence_early_stop(solve):
    pts = torch.tensor(EXAMPLE_POINTS)
    init = torch.tensor(EXAMPLE_INIT)
    labels, cent, iters = solve(pts, init, tol=10.0)
    assert iters == 1, iters
    assert torch.equal(cent, torch.tensor([[0.0, 1.0], [0.0, 2.0], [10.0, 1.0]])), \
        cent.tolist()
    assert torch.equal(labels, torch.tensor(EXAMPLE_LABELS)), labels.tolist()
    _assert_matches(solve, pts, init, tol=10.0)


def check_max_iters_truncation(solve):
    pts = torch.tensor(EXAMPLE_POINTS)
    init = torch.tensor(EXAMPLE_INIT)
    expect = {
        1: [[0.0, 1.0], [0.0, 2.0], [10.0, 1.0]],
        2: [[0.0, 0.0], [0.0, 2.0], [10.0, 1.0]],
        3: [[0.0, 0.0], [0.0, 2.0], [10.0, 1.0]],
    }
    for mi, c in expect.items():
        labels, cent, iters = solve(pts, init, max_iters=mi)
        assert iters == mi, (mi, iters)
        assert torch.equal(cent, torch.tensor(c)), (mi, cent.tolist())
        _assert_matches(solve, pts, init, max_iters=mi)
    # tol=0 且严格 <：位移已经为 0 也不提前停，跑满 max_iters
    _, cent0, it0 = solve(pts, init, tol=0.0, max_iters=5)
    assert it0 == 5, it0
    assert torch.equal(cent0, torch.tensor(EXAMPLE_CENTROIDS)), cent0.tolist()
    _assert_matches(solve, pts, init, tol=0.0, max_iters=5)


def check_tie_takes_smallest_index(solve):
    """两点关于两个重合质心对称：并列取小下标，空簇重播种后角色对调。"""
    pts = torch.tensor([[-1.0], [1.0]])
    init = torch.tensor([[0.0], [0.0]])
    labels, cent, iters = solve(pts, init)
    assert torch.equal(labels, torch.tensor([1, 0])), labels.tolist()
    assert torch.equal(cent, torch.tensor([[1.0], [-1.0]])), cent.tolist()
    assert iters == 3, iters
    _assert_matches(solve, pts, init)


def check_single_cluster_and_duplicates(solve):
    pts = torch.tensor(EXAMPLE_POINTS)
    labels, cent, iters = solve(pts, torch.tensor([[100.0, 100.0]]))
    assert torch.equal(labels, torch.zeros(4, dtype=torch.int64)), labels.tolist()
    assert torch.equal(cent, torch.tensor([[5.0, 1.0]])), cent.tolist()
    assert iters == 2, iters
    _assert_matches(solve, pts, torch.tensor([[100.0, 100.0]]))

    # N=K 且所有点重合：只有簇 0 非空，其余空簇按点下标依次重播种；
    # 所有质心数值相同 -> 首轮位移就是 0，立即收敛。
    dups = torch.zeros(4, 2)
    init = torch.zeros(4, 2)
    labels, cent, iters = solve(dups, init)
    assert torch.equal(labels, torch.zeros(4, dtype=torch.int64)), labels.tolist()
    assert torch.equal(cent, torch.zeros(4, 2)), cent.tolist()
    assert iters == 1, iters
    _assert_matches(solve, dups, init)


def check_oracle_grid(solve):
    cases = [
        (torch.tensor([[0.0, 0.0], [0.0, 2.0], [4.0, 0.0], [4.0, 2.0],
                       [8.0, 0.0], [8.0, 2.0]]),
         torch.tensor([[0.0, 0.0], [0.0, 0.0], [4.0, 0.0], [8.0, 0.0]]), 1e-4, 100),
        (torch.tensor([[0.0, 0.0], [0.0, 1.0], [10.0, 0.0], [10.0, 1.0],
                       [20.0, 0.0]]),
         torch.tensor([[0.0, 0.0], [0.0, 1.0], [20.0, 0.0]]), 1e-4, 100),
        (torch.tensor([[1.0], [2.0], [3.0], [10.0], [11.0]]),
         torch.tensor([[1.0], [1.0], [10.0]]), 1e-6, 100),
        (torch.tensor([[1.5, -2.0], [1.5, 2.0], [-3.0, 0.0], [4.0, 4.0]]),
         torch.tensor([[0.0, 0.0], [4.0, 4.0]]), 1e-4, 2),
    ]
    for pts, init, tol, mi in cases:
        _assert_matches(solve, pts, init, tol=tol, max_iters=mi)


def check_float64(solve):
    pts = torch.tensor([
        [0.125, 1.0], [0.5, 1.25], [2.0, 0.5], [2.5, 0.75], [8.0, 8.0],
    ], dtype=torch.float64)
    init = torch.tensor([[0.0, 1.0], [0.25, 1.0], [8.0, 8.0]], dtype=torch.float64)
    labels, cent, iters = solve(pts, init)
    assert cent.dtype == torch.float64
    assert labels.dtype == torch.int64
    exp_l, exp_c, exp_it = _oracle_kmeans(pts, init)
    assert torch.equal(labels, exp_l), (labels.tolist(), exp_l.tolist())
    assert torch.allclose(cent, exp_c, atol=1e-12, rtol=0), (cent.tolist(), exp_c.tolist())
    assert iters == exp_it
    _assert_matches(solve, pts, init)


def check_errors(solve):
    pts = torch.tensor(EXAMPLE_POINTS)
    init = torch.tensor(EXAMPLE_INIT)

    def bad(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    bad(lambda: solve(torch.zeros(0, 2), torch.zeros(1, 2)))
    bad(lambda: solve(pts, torch.zeros(0, 2)))
    bad(lambda: solve(torch.zeros(2, 2), torch.zeros(3, 2)))
    bad(lambda: solve(torch.zeros(2, 2), torch.zeros(1, 3)))
    bad(lambda: solve(torch.zeros(2, 2, 1), torch.zeros(1, 1)))
    bad(lambda: solve(torch.zeros(2, 2), torch.zeros(1, 1, 1)))
    bad(lambda: solve(pts.to(torch.float64), init))
    bad(lambda: solve(pts.to(torch.int64), init.to(torch.int64)))
    bad(lambda: solve(pts, init, max_iters=0))
    bad(lambda: solve(pts, init, max_iters=1.5))
    bad(lambda: solve(pts, init, tol=-1e-3))
