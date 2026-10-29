import math

import torch

# 示例期望值（实测回填）：L=7, D=2, mask=[1,1,0,1,1,1,0]
EXAMPLE_TOKEN_EMB = [
    [1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [2.0, 0.0],
    [0.0, 1.0], [1.0, 3.0], [0.0, 0.0],
]
EXAMPLE_MASK = [1, 1, 0, 1, 1, 1, 0]
EXAMPLE_BOUNDS = [[0, 3], [2, 5], [4, 6], [5, 7]]
EXAMPLE_HITS = [2, 0, 3, 2]
EXAMPLE_PARENT_SIZE = 2
EXAMPLE_CHUNK_VECS = [
    [0.4472135901451111, 0.8944271802902222],
    [0.8944271802902222, 0.4472135901451111],
    [0.24253563582897186, 0.9701425433158875],
    [0.3162277638912201, 0.9486832618713379],
]
EXAMPLE_PARENT_BOUNDS = [[0, 5], [4, 7]]
EXAMPLE_PARENT_VECS = [
    [0.7071068286895752, 0.7071068286895752],
    [0.24253563582897186, 0.9701425433158875],
]
EXAMPLE_HIT_TO_PARENT = [1, 0, 1, 1]


def _atol(dtype):
    return 1e-12 if dtype == torch.float64 else 1e-5


def _close(got, exp, dtype=None):
    dt = dtype or got.dtype
    return torch.allclose(got.to(dt), torch.as_tensor(exp, dtype=dt),
                          atol=_atol(dt), rtol=0)


def check_example(solve):
    emb = torch.tensor(EXAMPLE_TOKEN_EMB)
    mask = torch.tensor(EXAMPLE_MASK)
    bounds = torch.tensor(EXAMPLE_BOUNDS, dtype=torch.int64)
    hits = torch.tensor(EXAMPLE_HITS, dtype=torch.int64)
    cv, pb, pv, h2p = solve(emb, mask, bounds, hits, EXAMPLE_PARENT_SIZE)

    assert cv.shape == (4, 2) and cv.dtype == torch.float32, (cv.shape, cv.dtype)
    assert _close(cv, EXAMPLE_CHUNK_VECS), cv.tolist()
    assert torch.equal(pb, torch.tensor(EXAMPLE_PARENT_BOUNDS)), pb.tolist()
    assert pb.dtype == torch.int64
    assert pv.shape == (2, 2) and _close(pv, EXAMPLE_PARENT_VECS), pv.tolist()
    assert torch.equal(h2p, torch.tensor(EXAMPLE_HIT_TO_PARENT)), h2p.tolist()
    assert h2p.dtype == torch.int64


check_example.example = True


def _pool_rows(emb, mask, spans, dim, normalize):
    out = []
    for s, e in spans:
        acc = [0.0] * dim
        cnt = 0
        t = s
        while t < e:
            if mask[t] != 0:
                for k in range(dim):
                    acc[k] += float(emb[t][k])
                cnt += 1
            t += 1
        row = [v / cnt for v in acc] if cnt > 0 else [0.0] * dim
        if normalize:
            n = math.sqrt(sum(v * v for v in row))
            if n > 0:
                row = [v / n for v in row]
        out.append(row)
    return out


def _oracle(token_emb, attention_mask, chunk_bounds, hit_chunks, parent_size,
            normalize=True):
    """纯 Python 逐块/逐 token 累加均值，父块按 child // parent_size 分组重算。"""
    emb = token_emb.tolist()
    mask = attention_mask.tolist()
    bounds = [[int(s), int(e)] for s, e in chunk_bounds.tolist()]
    hits = [int(v) for v in hit_chunks.tolist()]
    dim = token_emb.shape[1]
    n_chunks = len(bounds)

    chunk_vecs = _pool_rows(emb, mask, bounds, dim, normalize)

    parent_ids = sorted({h // parent_size for h in hits})
    parent_spans = []
    for p in parent_ids:
        first = p * parent_size
        last = min((p + 1) * parent_size, n_chunks) - 1
        parent_spans.append([bounds[first][0], bounds[last][1]])
    parent_vecs = _pool_rows(emb, mask, parent_spans, dim, normalize)

    row_of = {p: i for i, p in enumerate(parent_ids)}
    mapping = [row_of[h // parent_size] for h in hits]

    dt = token_emb.dtype
    return (torch.tensor(chunk_vecs, dtype=dt).reshape(n_chunks, dim),
            torch.tensor(parent_spans, dtype=torch.int64).reshape(len(parent_spans), 2),
            torch.tensor(parent_vecs, dtype=dt).reshape(len(parent_spans), dim),
            torch.tensor(mapping, dtype=torch.int64))


def check_matches_oracle(solve):
    cases = [
        (torch.tensor(EXAMPLE_TOKEN_EMB), torch.tensor(EXAMPLE_MASK),
         torch.tensor(EXAMPLE_BOUNDS), torch.tensor(EXAMPLE_HITS), 2, True),
        (torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [2.0, 2.0, 0.0],
                       [-1.0, 0.0, 1.0], [0.0, -2.0, 1.0], [1.0, 1.0, 1.0],
                       [3.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 4.0, 2.0]]),
         torch.tensor([1, 0, 1, 1, 1, 0, 1, 0, 1]),
         torch.tensor([[0, 4], [3, 7], [6, 9]]),
         torch.tensor([0, 2, 2]), 2, True),
        (torch.tensor([[1.0, 1.0], [2.0, 0.0], [0.0, 3.0], [4.0, 4.0],
                       [0.0, 1.0], [1.0, 0.0], [0.0, 2.0], [3.0, 3.0],
                       [1.0, 1.0], [0.0, 5.0]]),
         torch.ones(10, dtype=torch.bool),
         torch.tensor([[0, 2], [4, 6], [8, 10]]),
         torch.tensor([1]), 3, True),
        (torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
         torch.tensor([1, 0, 1]),
         torch.tensor([[0, 0], [0, 3], [3, 3]]),
         torch.tensor([0, 2]), 2, True),
        (torch.zeros(0, 2), torch.zeros(0, dtype=torch.bool),
         torch.zeros(0, 2, dtype=torch.int64),
         torch.zeros(0, dtype=torch.int64), 3, True),
        (torch.tensor([[1.0, 1.0], [2.0, 0.0], [0.0, 3.0], [4.0, 4.0],
                       [0.0, 1.0], [1.0, 0.0]]),
         torch.tensor([1, 1, 0, 1, 1, 0]),
         torch.tensor([[1, 3], [3, 5], [4, 6]]),
         torch.tensor([2, 0, 2]), 10, True),
        (torch.tensor(EXAMPLE_TOKEN_EMB), torch.tensor(EXAMPLE_MASK),
         torch.tensor(EXAMPLE_BOUNDS), torch.tensor(EXAMPLE_HITS), 2, False),
    ]
    for k, (emb, mask, bounds, hits, ps, norm) in enumerate(cases):
        got = solve(emb, mask, bounds, hits, ps, normalize=norm)
        exp = _oracle(emb, mask, bounds, hits, ps, normalize=norm)
        atol = _atol(emb.dtype)
        assert got[0].shape == exp[0].shape, (k, got[0].shape, exp[0].shape)
        assert torch.allclose(got[0], exp[0], atol=atol, rtol=0), (k, got[0].tolist())
        assert torch.equal(got[1], exp[1]), (k, got[1].tolist(), exp[1].tolist())
        assert got[2].shape == exp[2].shape, (k, got[2].shape, exp[2].shape)
        assert torch.allclose(got[2], exp[2], atol=atol, rtol=0), (k, got[2].tolist())
        assert torch.equal(got[3], exp[3]), (k, got[3].tolist(), exp[3].tolist())
        assert got[0].dtype == emb.dtype and got[2].dtype == emb.dtype
        assert got[1].dtype == torch.int64 and got[3].dtype == torch.int64


def check_mask_participation(solve):
    # 均值只除有效 token 数：token1 被 mask 掉，(1+5)/2=3 而不是 4
    emb = torch.tensor([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [7.0, 0.0]])
    mask = torch.tensor([1, 0, 1, 0])
    cv, pb, pv, h2p = solve(emb, mask, torch.tensor([[0, 4]]),
                            torch.tensor([0]), 1, normalize=False)
    assert _close(cv, [[3.0, 0.0]]), cv.tolist()
    assert _close(pv, [[3.0, 0.0]]), pv.tolist()
    # 全被 mask 的块记 0（normalize 后仍是 0，不产生 NaN）
    emb2 = torch.tensor([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [7.0, 0.0]])
    cv, _, pv, _ = solve(emb2, torch.tensor([1, 0, 0, 0]),
                         torch.tensor([[0, 4], [1, 2]]), torch.tensor([0, 1]), 1)
    assert _close(cv, [[1.0, 0.0], [0.0, 0.0]]), cv.tolist()
    assert torch.isfinite(cv).all() and torch.isfinite(pv).all()
    assert _close(cv[1].norm(), [0.0]), cv[1].tolist()
    # 同样的块做原始均值：3 与 0
    cv, _, _, _ = solve(emb2, torch.tensor([1, 0, 0, 0]),
                        torch.tensor([[0, 4], [1, 2]]), torch.tensor([0, 1]), 1,
                        normalize=False)
    assert _close(cv, [[1.0, 0.0], [0.0, 0.0]]), cv.tolist()


def check_empty_hits(solve):
    emb = torch.tensor(EXAMPLE_TOKEN_EMB)
    mask = torch.tensor(EXAMPLE_MASK)
    bounds = torch.tensor(EXAMPLE_BOUNDS)
    hits = torch.zeros(0, dtype=torch.int64)
    cv, pb, pv, h2p = solve(emb, mask, bounds, hits, 2)
    assert _close(cv, EXAMPLE_CHUNK_VECS), cv.tolist()
    assert pb.shape == (0, 2) and pb.dtype == torch.int64
    assert pv.shape == (0, 2) and pv.dtype == torch.float32
    assert h2p.shape == (0,) and h2p.dtype == torch.int64
    # normalize=False 时空命中同样只影响父块输出
    cv2, pb2, pv2, h2p2 = solve(emb, mask, bounds, hits, 2, normalize=False)
    assert _close(cv2, [[0.5, 1.0], [1.0, 0.5], [0.5, 2.0], [1.0, 3.0]]), cv2.tolist()
    assert pb2.shape == (0, 2) and pv2.shape == (0, 2) and h2p2.shape == (0,)


def check_overlapping_parents_dedup(solve):
    # 子块滑窗重叠 -> 相邻父块区间也重叠（[0,6) 与 [5,10)）
    emb = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0],
                        [1.0, 1.0], [3.0, 0.0], [0.0, 3.0], [2.0, 2.0],
                        [1.0, 0.0], [0.0, 1.0]])
    mask = torch.tensor([1, 1, 1, 0, 1, 1, 1, 0, 1, 1])
    bounds = torch.tensor([[0, 4], [2, 6], [5, 9], [7, 10]])
    hits = torch.tensor([2, 0, 3, 1, 2])
    cv, pb, pv, h2p = solve(emb, mask, bounds, hits, 2)
    assert torch.equal(pb, torch.tensor([[0, 6], [5, 10]])), pb.tolist()
    assert pb[0, 1] > pb[1, 0], "父块区间应重叠"
    assert torch.equal(h2p, torch.tensor([1, 0, 1, 0, 1])), h2p.tolist()
    assert len(set(h2p.tolist())) == 2  # 5 次命中只产生 2 个父块
    exp = _oracle(emb, mask, bounds, hits, 2)
    assert torch.allclose(cv, exp[0], atol=1e-5, rtol=0), cv.tolist()
    assert torch.allclose(pv, exp[2], atol=1e-5, rtol=0), pv.tolist()
    # 同一父块内重复命中：只输出一次
    _, pb2, _, h2p2 = solve(emb, mask, bounds, torch.tensor([2, 2, 2]), 2)
    assert torch.equal(pb2, torch.tensor([[5, 10]])), pb2.tolist()
    assert torch.equal(h2p2, torch.tensor([0, 0, 0])), h2p2.tolist()
    # parent_size=1 时每个命中各自成父块，父块按下标升序输出
    _, pb3, _, h2p3 = solve(emb, mask, bounds, torch.tensor([3, 0, 2]), 1)
    assert torch.equal(pb3, torch.tensor([[0, 4], [5, 9], [7, 10]])), pb3.tolist()
    assert torch.equal(h2p3, torch.tensor([2, 0, 1])), h2p3.tolist()


def check_normalize_false(solve):
    emb = torch.tensor(EXAMPLE_TOKEN_EMB)
    mask = torch.tensor(EXAMPLE_MASK)
    bounds = torch.tensor(EXAMPLE_BOUNDS)
    hits = torch.tensor(EXAMPLE_HITS)
    cv, _, pv, _ = solve(emb, mask, bounds, hits, 2, normalize=False)
    assert _close(cv, [[0.5, 1.0], [1.0, 0.5], [0.5, 2.0], [1.0, 3.0]]), cv.tolist()
    assert _close(pv, [[0.75, 0.75], [0.5, 2.0]]), pv.tolist()
    # normalize=True：非零行范数为 1，零块保持 0
    emb2 = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    cv2, _, pv2, _ = solve(emb2, torch.tensor([1, 0]),
                           torch.tensor([[0, 1], [1, 2]]), torch.tensor([0, 1]), 2)
    assert _close(cv2, [[1.0, 0.0], [0.0, 0.0]]), cv2.tolist()
    assert _close(pv2, [[1.0, 0.0]]), pv2.tolist()
    assert abs(cv2[0].norm().item() - 1.0) < 1e-6
    assert cv2[1].norm().item() == 0.0
    assert torch.isfinite(cv2).all() and torch.isfinite(pv2).all()


def check_bounds_edges(solve):
    # 首块从 token 1 开始、含零长度块 [5,5) 与触尾块 [5,6)
    emb = torch.tensor([[9.0, 9.0], [1.0, 0.0], [0.0, 2.0], [2.0, 2.0],
                        [4.0, 0.0], [0.0, 4.0]])
    mask = torch.tensor([1, 1, 1, 0, 1, 1])
    bounds = torch.tensor([[1, 2], [2, 5], [5, 5], [5, 6]])
    hits = torch.tensor([0, 1, 2, 3])
    cv, pb, pv, h2p = solve(emb, mask, bounds, hits, 2)
    assert cv.shape == (4, 2)
    assert _close(cv[0], [[1.0, 0.0]]), cv[0].tolist()
    assert _close(cv[2], [[0.0, 0.0]]), cv[2].tolist()  # 零长度块
    assert abs(cv[2].norm().item()) == 0.0
    assert torch.equal(pb, torch.tensor([[1, 5], [5, 6]])), pb.tolist()
    assert torch.equal(h2p, torch.tensor([0, 0, 1, 1])), h2p.tolist()
    exp = _oracle(emb, mask, bounds, hits, 2)
    assert torch.allclose(cv, exp[0], atol=1e-5, rtol=0), cv.tolist()
    assert torch.allclose(pv, exp[2], atol=1e-5, rtol=0), pv.tolist()


def check_single_parent(solve):
    # parent_size 大于块数：所有命中落到唯一父块，区间覆盖首块 start 到末块 end
    emb = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 2.0],
                        [4.0, 0.0], [0.0, 4.0]])
    mask = torch.ones(6, dtype=torch.int64)
    bounds = torch.tensor([[1, 3], [3, 5], [4, 6]])
    hits = torch.tensor([2, 0, 2])
    cv, pb, pv, h2p = solve(emb, mask, bounds, hits, 10)
    assert torch.equal(pb, torch.tensor([[1, 6]])), pb.tolist()
    assert torch.equal(h2p, torch.tensor([0, 0, 0])), h2p.tolist()
    assert len(set(h2p.tolist())) == 1
    # 父块向量 = token 1..5 的均值（不是子块向量的均值）
    exp = _oracle(emb, mask, bounds, hits, 10)
    assert torch.allclose(pv, exp[2], atol=1e-5, rtol=0), pv.tolist()
    manual = (emb[1:6].sum(dim=0) / 5.0)
    manual = manual / manual.norm()
    assert torch.allclose(pv[0], manual, atol=1e-6, rtol=0), pv.tolist()


def check_dtype_float64(solve):
    emb = torch.tensor(EXAMPLE_TOKEN_EMB, dtype=torch.float64)
    mask = torch.tensor(EXAMPLE_MASK)
    bounds = torch.tensor(EXAMPLE_BOUNDS)
    hits = torch.tensor(EXAMPLE_HITS)
    got = solve(emb, mask, bounds, hits, 2)
    exp = _oracle(emb, mask, bounds, hits, 2)
    assert got[0].dtype == torch.float64 and got[2].dtype == torch.float64
    assert got[1].dtype == torch.int64 and got[3].dtype == torch.int64
    assert torch.allclose(got[0], exp[0], atol=1e-12, rtol=0), got[0].tolist()
    assert torch.allclose(got[2], exp[2], atol=1e-12, rtol=0), got[2].tolist()
    assert torch.equal(got[1], exp[1]) and torch.equal(got[3], exp[3])


def _expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def check_errors(solve):
    emb = torch.zeros(4, 2)
    mask = torch.ones(4, dtype=torch.int64)
    bounds = torch.tensor([[0, 2], [2, 4]])
    hits = torch.tensor([0])
    _expect_value_error(lambda: solve(torch.zeros(4), mask, bounds, hits, 1))
    _expect_value_error(lambda: solve(torch.zeros(4, 2, dtype=torch.long),
                                      mask, bounds, hits, 1))
    _expect_value_error(lambda: solve(emb, torch.ones(3), bounds, hits, 1))
    _expect_value_error(lambda: solve(emb, torch.ones(4, 1), bounds, hits, 1))
    _expect_value_error(lambda: solve(emb, mask, torch.zeros(2, 3), hits, 1))
    _expect_value_error(lambda: solve(emb, mask, bounds.float(), hits, 1))
    _expect_value_error(lambda: solve(emb, mask, torch.tensor([[0, 2], [2, 5]]),
                                      hits, 1))
    _expect_value_error(lambda: solve(emb, mask, torch.tensor([[0, 3], [2, 1]]),
                                      hits, 1))
    _expect_value_error(lambda: solve(emb, mask, torch.tensor([[2, 4], [0, 2]]),
                                      hits, 1))
    _expect_value_error(lambda: solve(emb, mask, bounds, torch.tensor([2]), 1))
    _expect_value_error(lambda: solve(emb, mask, bounds, torch.tensor([-1]), 1))
    _expect_value_error(lambda: solve(emb, mask, bounds,
                                      torch.tensor([[0]]), 1))
    _expect_value_error(lambda: solve(emb, mask, bounds,
                                      torch.tensor([0.0]), 1))
    _expect_value_error(lambda: solve(emb, mask, bounds, hits, 0))
    # C=0 时任何命中都越界；空命中合法
    empty_bounds = torch.zeros(0, 2, dtype=torch.int64)
    _expect_value_error(lambda: solve(emb, mask, empty_bounds,
                                      torch.tensor([0]), 1))
    cv, pb, pv, h2p = solve(emb, mask, empty_bounds,
                            torch.zeros(0, dtype=torch.int64), 1)
    assert cv.shape == (0, 2) and pb.shape == (0, 2)
    assert pv.shape == (0, 2) and h2p.shape == (0,)
