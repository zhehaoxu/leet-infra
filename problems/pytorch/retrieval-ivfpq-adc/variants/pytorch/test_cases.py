import torch

# 示例期望值（由参考实现实跑回填）：D=2, m=2, d=1, K=2, C=2
EXAMPLE_QUERIES = [[0.0, 0.0]]
EXAMPLE_CENTROIDS = [[0.0, 0.0], [10.0, 10.0]]
EXAMPLE_CODEBOOKS = [[[0.0], [10.0]], [[0.0], [10.0]]]
EXAMPLE_IDS = [[0, 1], [2, 3]]
EXAMPLE_CODES = [[[0, 0], [1, 0]], [[0, 1], [1, 1]]]
EXAMPLE_NPROBE = 2
EXAMPLE_K = 3
EXAMPLE_OUT_IDS = [[0, 1, 2]]
EXAMPLE_OUT_DISTS = [[0.0, 100.0, 100.0]]
EXAMPLE_NPROBE1_IDS = [[0, 1, -1]]
EXAMPLE_NPROBE1_DISTS = [[0.0, 100.0, float("inf")]]


def check_example(solve):
    q = torch.tensor(EXAMPLE_QUERIES, dtype=torch.float32)
    c = torch.tensor(EXAMPLE_CENTROIDS, dtype=torch.float32)
    cb = torch.tensor(EXAMPLE_CODEBOOKS, dtype=torch.float32)
    ids = [torch.tensor(x, dtype=torch.int64) for x in EXAMPLE_IDS]
    codes = [torch.tensor(x, dtype=torch.int64) for x in EXAMPLE_CODES]

    out_ids, out_d = solve(q, c, cb, ids, codes,
                           nprobe=EXAMPLE_NPROBE, k=EXAMPLE_K)
    exp_ids = torch.tensor(EXAMPLE_OUT_IDS, dtype=torch.int64)
    exp_d = torch.tensor(EXAMPLE_OUT_DISTS, dtype=torch.float32)
    assert torch.equal(out_ids, exp_ids), (out_ids.tolist(), exp_ids.tolist())
    assert torch.allclose(out_d, exp_d, atol=1e-5, rtol=1e-6), \
        (out_d.tolist(), exp_d.tolist())
    assert out_ids.shape == (1, 3) and out_ids.dtype == torch.int64
    assert out_d.shape == (1, 3) and out_d.dtype == torch.float32

    # nprobe=1 只扫最近的 cell 0（doc 0/1），k=3 时尾部补 -1 / +inf
    ids1, d1 = solve(q, c, cb, ids, codes, nprobe=1, k=3)
    exp_ids1 = torch.tensor(EXAMPLE_NPROBE1_IDS, dtype=torch.int64)
    exp_d1 = torch.tensor(EXAMPLE_NPROBE1_DISTS, dtype=torch.float32)
    assert torch.equal(ids1, exp_ids1), (ids1.tolist(), exp_ids1.tolist())
    assert torch.allclose(d1, exp_d1, atol=1e-5, rtol=1e-6), d1.tolist()


check_example.example = True


def _random_case(seed, qn=3, nc=4, m=2, d=3, ksub=5, maxn=6, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    dim = m * d
    queries = torch.randn(qn, dim, generator=g, dtype=dtype)
    centroids = torch.randn(nc, dim, generator=g, dtype=dtype)
    codebooks = torch.randn(m, ksub, d, generator=g, dtype=dtype)
    ids, codes = [], []
    nxt = 0
    for _ in range(nc):
        n = int(torch.randint(0, maxn + 1, (1,), generator=g).item())
        ids.append(torch.arange(nxt, nxt + n, dtype=torch.int64))
        nxt += n
        codes.append(torch.randint(0, ksub, (n, m), generator=g))
    return queries, centroids, codebooks, ids, codes


def _oracle(queries, centroids, codebooks, invlist_ids, invlist_codes,
            nprobe, k):
    """纯 Python 逐查询、逐 cell、逐文档、逐子空间重算 ADC 距离。"""
    ql = queries.tolist()
    cl = centroids.tolist()
    cbl = codebooks.tolist()
    ids_l = [t.tolist() for t in invlist_ids]
    codes_l = [t.tolist() for t in invlist_codes]
    qn, dim = queries.shape
    nc = centroids.shape[0]
    m, _, d = codebooks.shape
    p = min(nprobe, nc)
    out_ids = torch.full((qn, k), -1, dtype=torch.int64)
    out_d = torch.full((qn, k), float("inf"), dtype=queries.dtype)
    for qi in range(qn):
        cd = [sum((ql[qi][t] - cl[c][t]) ** 2 for t in range(dim))
              for c in range(nc)]
        order = sorted(range(nc), key=lambda c: (cd[c], c))[:p]
        cand = []
        for c in order:
            for i, doc in enumerate(ids_l[c]):
                dist = 0.0
                for j in range(m):
                    code = codes_l[c][i][j]
                    for t in range(d):
                        diff = ql[qi][j * d + t] - cbl[j][code][t]
                        dist += diff * diff
                cand.append((dist, doc))
        cand.sort(key=lambda x: (x[0], x[1]))
        for pos, (dist, doc) in enumerate(cand[:k]):
            out_ids[qi, pos] = doc
            out_d[qi, pos] = dist
    return out_ids, out_d


def _compare(solve, case, nprobe, k, atol, rtol):
    q, c, cb, ids, codes = case
    got_ids, got_d = solve(q, c, cb, ids, codes, nprobe=nprobe, k=k)
    exp_ids, exp_d = _oracle(q, c, cb, ids, codes, nprobe=nprobe, k=k)
    assert torch.equal(got_ids, exp_ids), \
        (nprobe, k, got_ids.tolist(), exp_ids.tolist())
    assert got_d.dtype == q.dtype, (got_d.dtype, q.dtype)
    assert got_d.shape == (q.shape[0], k), got_d.shape
    assert torch.allclose(got_d, exp_d, atol=atol, rtol=rtol), \
        (nprobe, k, got_d.tolist(), exp_d.tolist(),
         (got_d - exp_d).abs().max().item())


def check_matches_oracle_float32(solve):
    for seed in range(5):
        case = _random_case(10 + seed, dtype=torch.float32)
        for nprobe in (0, 1, 2, 4, 9):
            for k in (0, 1, 3, 10):
                _compare(solve, case, nprobe, k, 1e-5, 1e-6)


def check_matches_oracle_float64(solve):
    for seed in range(3):
        case = _random_case(50 + seed, dtype=torch.float64)
        for nprobe in (0, 1, 3, 4, 7):
            for k in (0, 2, 6):
                _compare(solve, case, nprobe, k, 1e-9, 1e-12)


def check_topk_truncation(solve):
    case = _random_case(77, qn=2, nc=3, m=2, d=2, ksub=6, maxn=8)
    q, c, cb, ids, codes = case
    got_ids, got_d = solve(q, c, cb, ids, codes, nprobe=3, k=2)
    exp_ids, exp_d = _oracle(q, c, cb, ids, codes, 3, 2)
    assert got_ids.shape == (2, 2) and got_d.shape == (2, 2)
    assert torch.equal(got_ids, exp_ids), (got_ids.tolist(), exp_ids.tolist())
    assert torch.allclose(got_d, exp_d, atol=1e-5, rtol=1e-6), got_d.tolist()
    # k 大于候选数：真实候选之后必须是 -1 / +inf
    wide_ids, wide_d = solve(q, c, cb, ids, codes, nprobe=3, k=40)
    for row in range(2):
        n_real = int((wide_ids[row] >= 0).sum().item())
        assert n_real == int((wide_d[row] < float("inf")).sum().item())
        assert bool((wide_ids[row, n_real:] == -1).all())
        assert bool(torch.isinf(wide_d[row, n_real:]).all())


def check_tie_smaller_global_id(solve):
    # 4 个码字到 query 的距离都恰好是 1，id 顺序故意打乱
    q = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    c = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    cb = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]],
                      dtype=torch.float32)
    ids = [torch.tensor([9, 2, 7, 4], dtype=torch.int64)]
    codes = [torch.tensor([[0], [1], [2], [3]], dtype=torch.int64)]
    out_ids, out_d = solve(q, c, cb, ids, codes, nprobe=1, k=4)
    assert torch.equal(out_ids, torch.tensor([[2, 4, 7, 9]], dtype=torch.int64)), \
        out_ids.tolist()
    assert torch.allclose(out_d, torch.ones(1, 4), atol=1e-6), out_d.tolist()
    # 距离相同再取小 id：k 截断也按同一顺序
    out_ids2, _ = solve(q, c, cb, ids, codes, nprobe=1, k=2)
    assert torch.equal(out_ids2, torch.tensor([[2, 4]], dtype=torch.int64)), \
        out_ids2.tolist()


def check_degenerate_cases(solve):
    q = torch.tensor(EXAMPLE_QUERIES, dtype=torch.float32)
    c = torch.tensor(EXAMPLE_CENTROIDS, dtype=torch.float32)
    cb = torch.tensor(EXAMPLE_CODEBOOKS, dtype=torch.float32)
    ids = [torch.tensor(x, dtype=torch.int64) for x in EXAMPLE_IDS]
    codes = [torch.tensor(x, dtype=torch.int64) for x in EXAMPLE_CODES]

    # k=0
    z_ids, z_d = solve(q, c, cb, ids, codes, nprobe=2, k=0)
    assert z_ids.shape == (1, 0) and z_d.shape == (1, 0)
    assert z_ids.dtype == torch.int64 and z_d.dtype == torch.float32

    # nprobe=0：即使倒排表非空也不探测
    ni, nd = solve(q, c, cb, ids, codes, nprobe=0, k=3)
    assert torch.equal(ni, torch.full((1, 3), -1, dtype=torch.int64)), ni.tolist()
    assert bool(torch.isinf(nd).all())

    # 所有 cell 都为空
    empty_ids = [torch.zeros(0, dtype=torch.int64)] * 2
    empty_codes = [torch.zeros(0, 2, dtype=torch.int64)] * 2
    ei, ed = solve(q, c, cb, empty_ids, empty_codes, nprobe=5, k=3)
    assert torch.equal(ei, torch.full((1, 3), -1, dtype=torch.int64)), ei.tolist()
    assert bool(torch.isinf(ed).all())

    # nprobe > C 等价于 C：显式比较
    a_ids, a_d = solve(q, c, cb, ids, codes, nprobe=2, k=4)
    b_ids, b_d = solve(q, c, cb, ids, codes, nprobe=99, k=4)
    assert torch.equal(a_ids, b_ids) and torch.allclose(a_d, b_d)

    # C=0：没有粗中心也没有倒排表
    c0 = torch.zeros((0, 2), dtype=torch.float32)
    ci, cd = solve(q, c0, cb, [], [], nprobe=3, k=2)
    assert torch.equal(ci, torch.full((1, 2), -1, dtype=torch.int64)), ci.tolist()
    assert bool(torch.isinf(cd).all())

    # Q=0
    qi, qd = solve(torch.zeros((0, 2)), c, cb, ids, codes, nprobe=2, k=3)
    assert qi.shape == (0, 3) and qd.shape == (0, 3)
    assert qi.dtype == torch.int64 and qd.dtype == torch.float32


def _expect_value_error(fn, tag):
    try:
        fn()
    except ValueError:
        return
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"{tag}: 期望 ValueError，实际 {type(exc).__name__}: {exc}")
    raise AssertionError(f"{tag}: 期望 ValueError，但没有抛错")


def check_nprobe_tie_smaller_cell(solve):
    # 4 个中心到 query 的平方距离都为 1：nprobe=2 必须选 cell 0/1（小号）
    q = torch.tensor([[0.0, 0.0]])
    c = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    cb = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]])
    empty = torch.zeros(0, dtype=torch.int64)
    codes_empty = torch.zeros(0, 1, dtype=torch.int64)
    # 只在 cell 2/3 放候选；若并列时错选大号 cell，就会错误地返回它们
    ids = [empty, empty,
           torch.tensor([20], dtype=torch.int64),
           torch.tensor([30], dtype=torch.int64)]
    codes = [codes_empty, codes_empty,
             torch.tensor([[0]], dtype=torch.int64),
             torch.tensor([[1]], dtype=torch.int64)]
    out_ids, out_d = solve(q, c, cb, ids, codes, nprobe=2, k=2)
    assert torch.equal(out_ids, torch.full((1, 2), -1, dtype=torch.int64)), \
        out_ids.tolist()
    assert bool(torch.isinf(out_d).all())
    out_ids4, out_d4 = solve(q, c, cb, ids, codes, nprobe=4, k=2)
    assert torch.equal(out_ids4, torch.tensor([[20, 30]], dtype=torch.int64)), \
        out_ids4.tolist()
    assert torch.allclose(out_d4, torch.ones(1, 2), atol=1e-6), out_d4.tolist()


def check_invalid_inputs(solve):
    q = torch.zeros(1, 2)
    c = torch.zeros(2, 2)
    cb = torch.zeros(2, 2, 1)          # m*d = 2 恰好匹配
    ids = [torch.tensor([0]), torch.tensor([1])]
    codes = [torch.tensor([[0, 0]]), torch.tensor([[0, 0]])]
    _expect_value_error(lambda: solve(q[0], c, cb, ids, codes), "queries 1-D")
    _expect_value_error(lambda: solve(q, c, torch.zeros(2, 2, 2), ids, codes),
                        "D != m*d")
    _expect_value_error(lambda: solve(q, c.double(), cb, ids, codes),
                        "dtype 不一致")
    _expect_value_error(lambda: solve(q, c, cb, ids[:1], codes),
                        "倒排表长度不匹配")
    _expect_value_error(lambda: solve(q, c, cb, [torch.tensor([0], dtype=torch.int32),
                                                 torch.tensor([1])], codes),
                        "ids 非 int64")
    _expect_value_error(lambda: solve(q, c, cb, [torch.tensor([-1]), torch.tensor([1])],
                                      codes), "负 id")
    bad_codes = [torch.tensor([[0, 2]]), torch.tensor([[0, 0]])]
    _expect_value_error(lambda: solve(q, c, cb, ids, bad_codes), "code 越界")
    _expect_value_error(lambda: solve(q, c, cb, ids, [torch.tensor([[0]]), codes[1]]),
                        "codes 第二维错误")
    _expect_value_error(lambda: solve(q, c, cb, ids, codes, nprobe=-1), "nprobe<0")
    _expect_value_error(lambda: solve(q, c, cb, ids, codes, k=-1), "k<0")
