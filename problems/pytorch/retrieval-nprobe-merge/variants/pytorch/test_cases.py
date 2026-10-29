import torch

# 示例期望值（由参考实现实跑回填）
EXAMPLE_QUERIES = [[0.0, 0.0]]
EXAMPLE_CENTROIDS = [[0.0, 0.0], [3.0, 0.0], [1.0, 0.0]]
EXAMPLE_VECTORS = [
    [[0.0, 0.0], [0.0, 0.0]],
    [[3.0, 0.0]],
    [[2.0, 0.0], [1.0, 0.0]],
]
EXAMPLE_IDS = [[7, 3], [8], [3, 5]]
EXAMPLE_NPROBE = 2
EXAMPLE_K = 3
EXAMPLE_OUT_IDS = [[3, 7, 5]]
EXAMPLE_OUT_DISTS = [[0.0, 0.0, 1.0]]
EXAMPLE_CELLS = [[0, 2]]
EXAMPLE_STATS = [[4, 3, 1, 3]]
EXAMPLE_NPROBE1_IDS = [[3, 7, -1]]
EXAMPLE_NPROBE1_DISTS = [[0.0, 0.0, float("inf")]]
EXAMPLE_NPROBE1_CELLS = [[0]]
EXAMPLE_NPROBE1_STATS = [[2, 2, 0, 2]]


def _example_inputs(dtype=torch.float32):
    q = torch.tensor(EXAMPLE_QUERIES, dtype=dtype)
    c = torch.tensor(EXAMPLE_CENTROIDS, dtype=dtype)
    vecs = [torch.tensor(x, dtype=dtype) for x in EXAMPLE_VECTORS]
    ids = [torch.tensor(x, dtype=torch.int64) for x in EXAMPLE_IDS]
    return q, c, vecs, ids


def check_example(solve):
    q, c, vecs, ids = _example_inputs()
    out_ids, out_d, cells, stats = solve(q, c, vecs, ids,
                                         nprobe=EXAMPLE_NPROBE, k=EXAMPLE_K)
    exp_ids = torch.tensor(EXAMPLE_OUT_IDS, dtype=torch.int64)
    exp_d = torch.tensor(EXAMPLE_OUT_DISTS, dtype=torch.float32)
    exp_cells = torch.tensor(EXAMPLE_CELLS, dtype=torch.int64)
    exp_stats = torch.tensor(EXAMPLE_STATS, dtype=torch.int64)
    assert torch.equal(out_ids, exp_ids), (out_ids.tolist(), exp_ids.tolist())
    assert torch.allclose(out_d, exp_d, atol=1e-5, rtol=1e-6), \
        (out_d.tolist(), exp_d.tolist())
    assert torch.equal(cells, exp_cells), (cells.tolist(), exp_cells.tolist())
    assert torch.equal(stats, exp_stats), (stats.tolist(), exp_stats.tolist())
    assert out_ids.shape == (1, 3) and out_ids.dtype == torch.int64
    assert out_d.shape == (1, 3) and out_d.dtype == torch.float32
    assert cells.dtype == torch.int64 and stats.dtype == torch.int64

    # nprobe=1 -> 只探测 cell 0，候选只有 id 3/7，k=3 尾部填充
    ids1, d1, cells1, stats1 = solve(q, c, vecs, ids, nprobe=1, k=3)
    assert torch.equal(ids1, torch.tensor(EXAMPLE_NPROBE1_IDS, dtype=torch.int64)), \
        ids1.tolist()
    exp_d1 = torch.tensor(EXAMPLE_NPROBE1_DISTS, dtype=torch.float32)
    assert torch.allclose(d1, exp_d1, atol=1e-5, rtol=1e-6), d1.tolist()
    assert torch.equal(cells1, torch.tensor(EXAMPLE_NPROBE1_CELLS,
                                            dtype=torch.int64)), cells1.tolist()
    assert torch.equal(stats1, torch.tensor(EXAMPLE_NPROBE1_STATS,
                                            dtype=torch.int64)), stats1.tolist()


check_example.example = True


def _random_case(seed, qn=3, nc=4, dim=3, maxn=6, pool=8, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    queries = torch.randn(qn, dim, generator=g, dtype=dtype)
    centroids = torch.randn(nc, dim, generator=g, dtype=dtype)
    vecs, ids = [], []
    for _ in range(nc):
        n = int(torch.randint(0, maxn + 1, (1,), generator=g).item())
        vecs.append(torch.randn(n, dim, generator=g, dtype=dtype))
        # 有放回抽 id：制造跨 cell / 同 cell 的重复
        ids.append(torch.randint(0, pool, (n,), generator=g))
    return queries, centroids, vecs, ids


def _oracle(queries, centroids, vecs, ids, nprobe, k):
    """纯 Python：逐 cell、逐 posting 求距离，dict 去重取 min，再显式排序。"""
    ql = queries.tolist()
    cl = centroids.tolist()
    vl = [t.tolist() for t in vecs]
    il = [t.tolist() for t in ids]
    qn, dim = queries.shape
    nc = centroids.shape[0]
    p = min(nprobe, nc)
    out_ids = torch.full((qn, k), -1, dtype=torch.int64)
    out_d = torch.full((qn, k), float("inf"), dtype=queries.dtype)
    out_cells = torch.zeros((qn, p), dtype=torch.int64)
    out_stats = torch.zeros((qn, 4), dtype=torch.int64)
    for qi in range(qn):
        cd = [sum((ql[qi][t] - cl[c][t]) ** 2 for t in range(dim))
              for c in range(nc)]
        order = sorted(range(nc), key=lambda c: (cd[c], c))[:p]
        for pos, c in enumerate(order):
            out_cells[qi, pos] = c
        best = {}
        n_post = 0
        for c in order:
            for i, doc in enumerate(il[c]):
                n_post += 1
                dist = sum((ql[qi][t] - vl[c][i][t]) ** 2 for t in range(dim))
                if doc not in best or dist < best[doc]:
                    best[doc] = dist
        items = sorted(best.items(), key=lambda kv: (kv[1], kv[0]))
        t = min(k, len(items))
        for pos, (doc, dist) in enumerate(items[:t]):
            out_ids[qi, pos] = doc
            out_d[qi, pos] = dist
        out_stats[qi] = torch.tensor(
            [n_post, len(best), n_post - len(best), t], dtype=torch.int64)
    return out_ids, out_d, out_cells, out_stats


def _compare(solve, case, nprobe, k, atol, rtol):
    q, c, vecs, ids = case
    got = solve(q, c, vecs, ids, nprobe=nprobe, k=k)
    exp = _oracle(q, c, vecs, ids, nprobe=nprobe, k=k)
    names = ("ids", "dists", "cells", "stats")
    for name, g, e in zip(names, got, exp):
        if name in ("ids", "cells", "stats"):
            assert torch.equal(g, e), \
                (name, nprobe, k, g.tolist(), e.tolist())
        else:
            assert g.dtype == q.dtype, (g.dtype, q.dtype)
            assert torch.allclose(g, e, atol=atol, rtol=rtol), \
                (name, nprobe, k, g.tolist(), e.tolist())
    assert got[0].shape == (q.shape[0], k)
    assert got[2].shape == (q.shape[0], min(nprobe, c.shape[0]))
    assert got[3].shape == (q.shape[0], 4)


def check_matches_oracle_float32(solve):
    for seed in range(5):
        case = _random_case(10 + seed, dtype=torch.float32)
        for nprobe in (0, 1, 2, 4, 9):
            for k in (0, 1, 3, 10):
                _compare(solve, case, nprobe, k, 1e-5, 1e-6)


def check_matches_oracle_float64(solve):
    for seed in range(3):
        case = _random_case(60 + seed, dtype=torch.float64)
        for nprobe in (0, 1, 3, 4, 8):
            for k in (0, 2, 6):
                _compare(solve, case, nprobe, k, 1e-9, 1e-12)


def check_dedup_keeps_min(solve):
    q, c, vecs, ids = _example_inputs()
    out_ids, out_d, cells, stats = solve(q, c, vecs, ids, nprobe=2, k=3)
    # id=3 在 cell 0（距离 0）与 cell 2（距离 4）各出现一次，只保留最小距离
    assert torch.equal(out_ids, torch.tensor([[3, 7, 5]], dtype=torch.int64)), \
        out_ids.tolist()
    assert torch.allclose(out_d, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-6), \
        out_d.tolist()
    assert torch.equal(stats, torch.tensor([[4, 3, 1, 3]], dtype=torch.int64)), \
        stats.tolist()

    # 同一 cell 内也有重复 id：距离 9 / 0 / 1，去重后只剩 id=1、距离 0
    q1 = torch.tensor([[0.0, 0.0]])
    c1 = torch.tensor([[0.0, 0.0]])
    v1 = [torch.tensor([[3.0, 0.0], [0.0, 0.0], [0.0, 1.0]])]
    i1 = [torch.tensor([1, 1, 1], dtype=torch.int64)]
    oi, od, oc, ost = solve(q1, c1, v1, i1, nprobe=1, k=2)
    assert torch.equal(oi, torch.tensor([[1, -1]], dtype=torch.int64)), oi.tolist()
    assert torch.allclose(od, torch.tensor([[0.0, float("inf")]]), atol=1e-6), \
        od.tolist()
    assert torch.equal(ost, torch.tensor([[3, 1, 2, 1]], dtype=torch.int64)), \
        ost.tolist()


def check_probe_order_and_ties(solve):
    # 到中心距离：c0=4, c1=1, c2=1, c3=1 -> 探测顺序 [1, 2, 3, 0]（并列取小 cell 号）
    q = torch.tensor([[0.0, 0.0]])
    c = torch.tensor([[0.0, 2.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    empty_v = [torch.zeros((0, 2)) for _ in range(4)]
    empty_i = [torch.zeros(0, dtype=torch.int64) for _ in range(4)]
    _, _, cells, stats = solve(q, c, empty_v, empty_i, nprobe=4, k=3)
    assert torch.equal(cells, torch.tensor([[1, 2, 3, 0]], dtype=torch.int64)), \
        cells.tolist()
    assert torch.equal(stats, torch.zeros((1, 4), dtype=torch.int64)), stats.tolist()
    # nprobe=2 时探测前两个：1 和 2（不是按 cell 号取 0,1）
    _, _, cells2, _ = solve(q, c, empty_v, empty_i, nprobe=2, k=3)
    assert torch.equal(cells2, torch.tensor([[1, 2]], dtype=torch.int64)), \
        cells2.tolist()
    # nprobe=1 只取最近且 cell 号最小者
    _, _, cells1, _ = solve(q, c, empty_v, empty_i, nprobe=1, k=3)
    assert torch.equal(cells1, torch.tensor([[1]], dtype=torch.int64)), \
        cells1.tolist()


def check_nprobe_boundaries(solve):
    case = _random_case(88, qn=2, nc=5, dim=4, maxn=5, pool=6)
    q, c, vecs, ids = case
    for npb in (0, 1, 5, 11):
        got = solve(q, c, vecs, ids, nprobe=npb, k=4)
        exp = _oracle(q, c, vecs, ids, npb, 4)
        for name, g, e in zip(("ids", "dists", "cells", "stats"), got, exp):
            if name == "dists":
                assert torch.allclose(g, e, atol=1e-5, rtol=1e-6), \
                    (npb, g.tolist(), e.tolist())
            else:
                assert torch.equal(g, e), (name, npb, g.tolist(), e.tolist())
    # stats 第 1 列 = 所有被探测 cell 的 posting 总数
    _, _, _, st = solve(q, c, vecs, ids, nprobe=5, k=4)
    n_all = sum(int(t.numel()) for t in ids)
    assert int(st[0, 0]) == n_all, (st.tolist(), n_all)
    # nprobe 超界时 cells 的行就是全部 cell 的完整顺序
    _, _, cells_all, _ = solve(q, c, vecs, ids, nprobe=11, k=4)
    assert cells_all.shape == (2, 5)


def check_topk_truncation(solve):
    case = _random_case(99, qn=2, nc=3, dim=2, maxn=8, pool=20)
    q, c, vecs, ids = case
    got = solve(q, c, vecs, ids, nprobe=3, k=2)
    exp = _oracle(q, c, vecs, ids, 3, 2)
    assert got[0].shape == (2, 2) and got[1].shape == (2, 2)
    assert torch.equal(got[0], exp[0]), (got[0].tolist(), exp[0].tolist())
    assert torch.allclose(got[1], exp[1], atol=1e-5, rtol=1e-6), got[1].tolist()
    assert int(got[3][0, 3]) == 2 and int(got[3][1, 3]) == 2, got[3].tolist()
    # k 大于候选数：尾部 -1 / +inf
    wide = solve(q, c, vecs, ids, nprobe=3, k=30)
    for row in range(2):
        n_real = int((wide[0][row] >= 0).sum().item())
        assert bool((wide[0][row, n_real:] == -1).all())
        assert bool(torch.isinf(wide[1][row, n_real:]).all())
        assert int(wide[3][row, 3]) == n_real


def check_empty_and_k0(solve):
    q = torch.tensor(EXAMPLE_QUERIES)
    c = torch.tensor(EXAMPLE_CENTROIDS)
    empty_v = [torch.zeros((0, 2)) for _ in range(3)]
    empty_i = [torch.zeros(0, dtype=torch.int64) for _ in range(3)]
    oi, od, cells, stats = solve(q, c, empty_v, empty_i, nprobe=3, k=4)
    assert torch.equal(oi, torch.full((1, 4), -1, dtype=torch.int64)), oi.tolist()
    assert bool(torch.isinf(od).all())
    assert cells.shape == (1, 3)
    assert torch.equal(stats, torch.zeros((1, 4), dtype=torch.int64)), stats.tolist()

    # k=0：结果空，但 cells / stats 仍按探测结果填写
    q2, c2, v2, i2 = _example_inputs()
    zi, zd, zc, zs = solve(q2, c2, v2, i2, nprobe=2, k=0)
    assert zi.shape == (1, 0) and zd.shape == (1, 0)
    assert torch.equal(zc, torch.tensor([[0, 2]], dtype=torch.int64)), zc.tolist()
    assert torch.equal(zs, torch.tensor([[4, 3, 1, 0]], dtype=torch.int64)), zs.tolist()

    # C=0：没有 cell 可探测
    c0 = torch.zeros((0, 2))
    ci, cd, cc, cs = solve(q, c0, [], [], nprobe=3, k=2)
    assert ci.shape == (1, 2) and cc.shape == (1, 0)
    assert torch.equal(ci, torch.full((1, 2), -1, dtype=torch.int64)), ci.tolist()
    assert bool(torch.isinf(cd).all())
    assert torch.equal(cs, torch.zeros((1, 4), dtype=torch.int64)), cs.tolist()

    # Q=0
    qi, qd, qc, qs = solve(torch.zeros((0, 2)), c2, v2, i2, nprobe=2, k=3)
    assert qi.shape == (0, 3) and qd.shape == (0, 3)
    assert qc.shape == (0, 2) and qs.shape == (0, 4)


def _expect_value_error(fn, tag):
    try:
        fn()
    except ValueError:
        return
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"{tag}: 期望 ValueError，实际 {type(exc).__name__}: {exc}")
    raise AssertionError(f"{tag}: 期望 ValueError，但没有抛错")


def check_invalid_inputs(solve):
    q = torch.zeros(1, 2)
    c = torch.zeros(2, 2)
    vecs = [torch.zeros(1, 2), torch.zeros(1, 2)]
    ids = [torch.tensor([0]), torch.tensor([1])]
    _expect_value_error(lambda: solve(q[0], c, vecs, ids), "queries 1-D")
    _expect_value_error(lambda: solve(q, c.double(), vecs, ids), "dtype 不一致")
    _expect_value_error(lambda: solve(q, c, vecs[:1], ids), "vectors 长度不匹配")
    _expect_value_error(lambda: solve(q, c, [torch.zeros(1, 3), vecs[1]], ids),
                        "vectors 维度错误")
    _expect_value_error(lambda: solve(q, c, vecs, [torch.tensor([0], dtype=torch.int32),
                                                   ids[1]]), "ids 非 int64")
    _expect_value_error(lambda: solve(q, c, vecs, [torch.tensor([-1]), ids[1]]),
                        "负 id")
    _expect_value_error(lambda: solve(q, c, [vecs[0].double(), vecs[1]], ids),
                        "vectors dtype 不一致")
    _expect_value_error(lambda: solve(q, c, vecs, ids, nprobe=-1), "nprobe<0")
    _expect_value_error(lambda: solve(q, c, vecs, ids, k=-1), "k<0")
