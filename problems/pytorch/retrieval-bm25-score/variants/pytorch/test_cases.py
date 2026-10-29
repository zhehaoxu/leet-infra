import math

import torch

# 示例期望值（实测回填）：N=3，doc_lens=[4,3,2]，query=[2,2,4]（2 出现两次）
EXAMPLE_DOCS = [[1, 2, 2, 3], [2, 3, 4, 0], [1, 4, -1, -1]]
EXAMPLE_LENS = [4, 3, 2]
EXAMPLE_QUERY = [2, 2, 4]
EXAMPLE_SCORES_LUCENE = [1.1817234106749925, 1.4100108877372068,
                         0.5442147286003255]
EXAMPLE_TOPK_LUCENE = [1, 0, 2]
EXAMPLE_SCORES_ROBERTSON = [-1.2843615683259197, -1.5324768712979722,
                            -0.5914823012027262]
EXAMPLE_TOPK_ROBERTSON = [2, 0, 1]


def _docs(rows):
    return torch.tensor(rows, dtype=torch.int64)


def _oracle_bm25(docs, doc_lens, query, k, k1, b, idf_mode):
    """纯 Python 双层循环暴力重算：逐文档扫 token、逐 query 出现累加。"""
    rows = docs.tolist()
    dl = doc_lens.tolist()
    q = query.tolist()
    n = len(dl)
    toks = [rows[d][:dl[d]] for d in range(n)]
    avgdl = sum(dl) / n
    scores = [0.0] * n
    for t in q:
        df = 0
        for d in range(n):
            for x in toks[d]:
                if x == t:
                    df += 1
                    break
        if idf_mode == "lucene":
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        else:
            idf = math.log((n - df + 0.5) / (df + 0.5))
        for d in range(n):
            tf = 0
            for x in toks[d]:
                if x == t:
                    tf += 1
            if tf == 0:
                continue
            if avgdl > 0.0:
                norm = k1 * (1.0 - b + b * dl[d] / avgdl)
            else:
                norm = k1 * (1.0 - b)
            scores[d] += idf * tf * (k1 + 1.0) / (tf + norm)
    order = sorted(range(n), key=lambda d: (-scores[d], d))[:min(k, n)]
    return scores, order


def _check_case(solve, rows, lens, query, k, k1=1.2, b=0.75,
                idf_mode="lucene"):
    docs = _docs(rows)
    dl = torch.tensor(lens, dtype=torch.int64)
    q = torch.tensor(query, dtype=torch.int64)
    got_s, got_t = solve(docs, dl, q, k=k, k1=k1, b=b, idf_mode=idf_mode)
    exp_s, exp_t = _oracle_bm25(docs, dl, q, k, k1, b, idf_mode)
    assert got_s.dtype == torch.float64, got_s.dtype
    assert got_t.dtype == torch.int64, got_t.dtype
    assert got_s.shape == (len(lens),), got_s.shape
    assert torch.allclose(got_s, torch.tensor(exp_s, dtype=torch.float64),
                          atol=1e-9, rtol=0.0), \
        (got_s.tolist(), exp_s)
    assert torch.equal(got_t, torch.tensor(exp_t, dtype=torch.int64)), \
        (got_t.tolist(), exp_t)


def check_example(solve):
    docs = _docs(EXAMPLE_DOCS)
    lens = torch.tensor(EXAMPLE_LENS, dtype=torch.int64)
    q = torch.tensor(EXAMPLE_QUERY, dtype=torch.int64)
    s, t = solve(docs, lens, q, k=3, idf_mode="lucene")
    assert torch.allclose(s, torch.tensor(EXAMPLE_SCORES_LUCENE,
                                          dtype=torch.float64), atol=1e-9), \
        s.tolist()
    assert torch.equal(t, torch.tensor(EXAMPLE_TOPK_LUCENE, dtype=torch.int64)), \
        t.tolist()
    s, t = solve(docs, lens, q, k=3, idf_mode="robertson")
    assert torch.allclose(s, torch.tensor(EXAMPLE_SCORES_ROBERTSON,
                                          dtype=torch.float64), atol=1e-9), \
        s.tolist()
    assert torch.equal(t, torch.tensor(EXAMPLE_TOPK_ROBERTSON, dtype=torch.int64)), \
        t.tolist()
    assert s.dtype == torch.float64 and t.dtype == torch.int64


check_example.example = True


def check_matches_bruteforce(solve):
    # 多组混合：重复 query 词、零长文档、自定义 k1/b、两种 idf
    _check_case(
        solve,
        [[1, 2, 3, 4, 5, 0, 0, 0],
         [2, 2, 4, 4, 1, 0, 0, 0],
         [3, 3, 3, 1, 2, 5, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [4, 5, 1, 2, 3, 3, 4, 4],
         [5, 5, 5, 5, 5, 1, 2, 3]],
        [5, 5, 6, 0, 8, 8],
        [1, 1, 3, 2], 4)
    _check_case(
        solve,
        [[1, 2, 3, 4, 5, 0, 0, 0],
         [2, 2, 4, 4, 1, 0, 0, 0],
         [3, 3, 3, 1, 2, 5, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [4, 5, 1, 2, 3, 3, 4, 4],
         [5, 5, 5, 5, 5, 1, 2, 3]],
        [5, 5, 6, 0, 8, 8],
        [1, 1, 3, 2], 10, k1=0.9, b=0.3, idf_mode="robertson")
    _check_case(solve, [[0, 0, 1], [2, 2, 2], [1, 2, 0], [1, 1, 1], [0, 2, 1]],
                [3, 3, 3, 3, 3], [0, 0, 0, 2], 2)
    _check_case(solve, [[7, 7, 7]], [3], [7, 7], 5)
    _check_case(solve, [[4, 4, 5], [4, 5, 5], [5, 5, 5], [4, 4, 4]],
                [3, 3, 3, 3], [4, 5], 3, k1=2.0, b=0.0)


def check_duplicate_query_accumulates(solve):
    docs = _docs(EXAMPLE_DOCS)
    lens = torch.tensor(EXAMPLE_LENS, dtype=torch.int64)
    s1, _ = solve(docs, lens, torch.tensor([2]), k=3)
    s2, _ = solve(docs, lens, torch.tensor([2, 2]), k=3)
    s4, _ = solve(docs, lens, torch.tensor([4]), k=3)
    s24, _ = solve(docs, lens, torch.tensor([2, 4]), k=3)
    s224, _ = solve(docs, lens, torch.tensor([2, 2, 4]), k=3)
    assert torch.allclose(s2, 2.0 * s1, atol=1e-12, rtol=0), s2.tolist()
    assert torch.allclose(s224, s2 + s4, atol=1e-12, rtol=0), s224.tolist()
    assert torch.allclose(s224, s24 + s1, atol=1e-12, rtol=0), s224.tolist()
    # 去重实现会得到 s24，必然与 s224 不同（示例中 2 的 df=2）
    assert not torch.allclose(s224, s24, atol=1e-9), s224.tolist()


def check_robertson_negative(solve):
    # N=3、df(2)=df(4)=2 > N/2 -> robertson idf 为负，分数整体变负
    docs = _docs(EXAMPLE_DOCS)
    lens = torch.tensor(EXAMPLE_LENS, dtype=torch.int64)
    q = torch.tensor(EXAMPLE_QUERY, dtype=torch.int64)
    sl, tl = solve(docs, lens, q, k=3, idf_mode="lucene")
    sr, tr = solve(docs, lens, q, k=3, idf_mode="robertson")
    assert bool((sl > 0).all()) and float(sl.min()) > 0.0
    assert bool((sr < 0).all()) and float(sr.max()) < 0.0
    # 负 idf 不截断：robertson 与 lucene 分数严格成同一负比例
    ratio = math.log(0.6) / math.log(1.6)
    assert torch.allclose(sr, ratio * sl, atol=1e-12, rtol=0), sr.tolist()
    assert tl.tolist() == [1, 0, 2] and tr.tolist() == [2, 0, 1]


def check_empty_query(solve):
    docs = _docs(EXAMPLE_DOCS)
    lens = torch.tensor(EXAMPLE_LENS, dtype=torch.int64)
    empty = torch.zeros(0, dtype=torch.int64)
    s, t = solve(docs, lens, empty, k=2)
    assert torch.equal(s, torch.zeros(3, dtype=torch.float64)), s.tolist()
    assert torch.equal(t, torch.tensor([0, 1], dtype=torch.int64)), t.tolist()
    s, t = solve(docs, lens, empty, k=10)
    assert torch.equal(t, torch.tensor([0, 1, 2], dtype=torch.int64))
    s, t = solve(docs, lens, empty, k=0)
    assert s.shape == (3,) and t.shape == (0,) and t.dtype == torch.int64


def check_oov_query(solve):
    docs = _docs(EXAMPLE_DOCS)
    lens = torch.tensor(EXAMPLE_LENS, dtype=torch.int64)
    q = torch.tensor([99, 99])
    for mode in ("lucene", "robertson"):
        s, t = solve(docs, lens, q, k=3, idf_mode=mode)
        assert torch.equal(s, torch.zeros(3, dtype=torch.float64)), (mode, s.tolist())
        assert torch.equal(t, torch.tensor([0, 1, 2], dtype=torch.int64)), \
            (mode, t.tolist())


def check_ties_ascending(solve):
    docs = _docs([[5, 5], [5, 5], [5, 5]])
    lens = torch.tensor([2, 2, 2], dtype=torch.int64)
    q = torch.tensor([5])
    s, t = solve(docs, lens, q, k=2)
    assert torch.allclose(s, s[0].expand(3), atol=0), s.tolist()
    assert torch.equal(t, torch.tensor([0, 1], dtype=torch.int64)), t.tolist()
    _, t = solve(docs, lens, q, k=10)
    assert torch.equal(t, torch.tensor([0, 1, 2], dtype=torch.int64)), t.tolist()


def check_all_empty_docs(solve):
    docs = torch.zeros((3, 0), dtype=torch.int64)
    lens = torch.zeros(3, dtype=torch.int64)
    s, t = solve(docs, lens, torch.tensor([1, 2]), k=3)
    assert torch.equal(s, torch.zeros(3, dtype=torch.float64)), s.tolist()
    assert not bool(torch.isnan(s).any())
    assert torch.equal(t, torch.tensor([0, 1, 2], dtype=torch.int64)), t.tolist()


def check_k_truncation(solve):
    docs = _docs(EXAMPLE_DOCS)
    lens = torch.tensor(EXAMPLE_LENS, dtype=torch.int64)
    q = torch.tensor(EXAMPLE_QUERY, dtype=torch.int64)
    s, t = solve(docs, lens, q, k=1)
    assert t.tolist() == [1], t.tolist()
    _, t = solve(docs, lens, q, k=0)
    assert t.shape == (0,)
    _, t = solve(docs, lens, q, k=100)
    assert t.tolist() == [1, 0, 2], t.tolist()
    assert s.shape == (3,)


def check_invalid_inputs(solve):
    good_docs = _docs(EXAMPLE_DOCS)
    good_lens = torch.tensor(EXAMPLE_LENS, dtype=torch.int64)
    good_q = torch.tensor(EXAMPLE_QUERY, dtype=torch.int64)

    def expect(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    expect(lambda: solve(torch.tensor([1, 2, 3]), good_lens, good_q))
    expect(lambda: solve(good_docs.to(torch.float32), good_lens, good_q))
    expect(lambda: solve(good_docs, good_lens.unsqueeze(0), good_q))
    expect(lambda: solve(good_docs, good_lens[:2], good_q))
    expect(lambda: solve(good_docs, torch.tensor([4, 3, 5]), good_q))
    expect(lambda: solve(good_docs, torch.tensor([-1, 3, 2]), good_q))
    expect(lambda: solve(good_docs, good_lens, good_q, k=-1))
    expect(lambda: solve(good_docs, good_lens, good_q, k1=0.0))
    expect(lambda: solve(good_docs, good_lens, good_q, b=1.5))
    expect(lambda: solve(good_docs, good_lens, good_q, idf_mode="bm25"))
    bad_docs = _docs([[1, -2, 0, 0], [1, 2, 0, 0], [1, 2, 0, 0]])
    expect(lambda: solve(bad_docs, torch.tensor([2, 2, 2]), good_q))
    expect(lambda: solve(good_docs, good_lens, torch.tensor([2, -2])))
    expect(lambda: solve(torch.zeros((0, 4), dtype=torch.int64),
                         torch.zeros(0, dtype=torch.int64), good_q))
