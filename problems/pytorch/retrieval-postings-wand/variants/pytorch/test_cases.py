import math

import torch

# 示例期望值（实测回填）：N=6，doc_lens 全为 4，block_size=2，k=2
EXAMPLE_LENS = [4, 4, 4, 4, 4, 4]
EXAMPLE_POSTINGS = [
    [[0, 2], [1, 1], [2, 1], [3, 1], [4, 1], [5, 2]],
    [[0, 1], [5, 3]],
    [[0, 2], [5, 2]],
]
EXAMPLE_BLOCK_SIZE = 2
EXAMPLE_K = 2
EXAMPLE_CANDIDATES = [0, 5]
EXAMPLE_TOPK = [5, 0]
EXAMPLE_PRUNED = 1
EXAMPLE_SCORES = [2.5472445775166186, 0.07410797215372183,
                  0.07410797215372183, 0.07410797215372183,
                  0.07410797215372183, 3.135598530191566]


def _t(rows):
    return torch.tensor(rows, dtype=torch.int64)


def _oracle(postings, doc_lens, k, k1=1.2, b=0.75):
    """纯 Python 暴力：逐文档逐 term 扫 postings 全打分，再排序。"""
    n = doc_lens.numel()
    dl = doc_lens.tolist()
    rows = [p.tolist() for p in postings]
    avgdl = sum(dl) / n
    score = [0.0] * n
    for r in rows:
        df = len(r)
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        for d, tf in r:
            if avgdl > 0.0:
                norm = k1 * (1.0 - b + b * dl[d] / avgdl)
            else:
                norm = k1 * (1.0 - b)
            score[d] += idf * tf * (k1 + 1.0) / (tf + norm)
    if not rows:
        cand = []
    else:
        cand = []
        for d in range(n):
            hit = True
            for r in rows:
                found = False
                for docid, _ in r:
                    if docid == d:
                        found = True
                        break
                if not found:
                    hit = False
                    break
            if hit:
                cand.append(d)
    union = set()
    for r in rows:
        for docid, _ in r:
            union.add(docid)
    order = sorted(union, key=lambda d: (-score[d], d))[:min(k, len(union))]
    return cand, order, score


def _check_case(solve, postings, lens, k, block_size=4, k1=1.2, b=0.75):
    dl = torch.tensor(lens, dtype=torch.int64)
    ps = [_t(p) for p in postings]
    got_c, got_t, got_p = solve(ps, dl, k=k, k1=k1, b=b, block_size=block_size)
    exp_c, exp_t, _ = _oracle(ps, dl, k, k1, b)
    assert got_c.dtype == torch.int64 and got_c.dim() == 1, got_c.dtype
    assert got_t.dtype == torch.int64, got_t.dtype
    assert isinstance(got_p, int) and got_p >= 0, got_p
    assert torch.equal(got_c, torch.tensor(exp_c, dtype=torch.int64)), \
        (got_c.tolist(), exp_c)
    assert torch.equal(got_t, torch.tensor(exp_t, dtype=torch.int64)), \
        (got_t.tolist(), exp_t)
    return got_p


def check_example(solve):
    dl = torch.tensor(EXAMPLE_LENS, dtype=torch.int64)
    ps = [_t(p) for p in EXAMPLE_POSTINGS]
    cand, topk, pruned = solve(ps, dl, k=EXAMPLE_K, block_size=EXAMPLE_BLOCK_SIZE)
    assert torch.equal(cand, torch.tensor(EXAMPLE_CANDIDATES, dtype=torch.int64)), \
        cand.tolist()
    assert torch.equal(topk, torch.tensor(EXAMPLE_TOPK, dtype=torch.int64)), \
        topk.tolist()
    assert pruned == EXAMPLE_PRUNED, pruned


check_example.example = True


def check_matches_bruteforce(solve):
    got = _check_case(
        solve,
        [[[0, 2], [2, 1], [4, 3], [6, 1]],
         [[1, 1], [2, 4], [5, 2], [6, 1]],
         [[2, 1], [6, 5]]],
        [5, 0, 3, 7, 2, 4, 6], 3, block_size=2)
    assert got >= 0
    _check_case(
        solve,
        [[[0, 1], [1, 5], [2, 2], [3, 4]],
         [[0, 3], [2, 1], [3, 1]],
         [[1, 2], [3, 6]]],
        [2, 3, 4, 5], 100, block_size=1)
    _check_case(
        solve,
        [[[0, 3], [1, 1], [2, 2], [3, 1], [4, 4]]],
        [4, 4, 4, 4, 4], 2, block_size=3, k1=2.0, b=0.2)
    _check_case(
        solve,
        [[[0, 1]], [[0, 1]]],
        [6], 1, block_size=4)
    _check_case(
        solve,
        [[[0, 1], [1, 2], [2, 3]],
         [[1, 1], [2, 1], [3, 2]]],
        [0, 0, 0, 0], 4, block_size=2)
    _check_case(
        solve,
        [[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]],
        [1, 2, 3, 4, 5, 6, 7], 5, block_size=2, k1=0.6, b=1.0)


def check_pruning_positive(solve):
    dl = torch.tensor(EXAMPLE_LENS, dtype=torch.int64)
    ps = [_t(p) for p in EXAMPLE_POSTINGS]
    cand, topk, pruned = solve(ps, dl, k=2, block_size=2)
    _, exp_topk, _ = _oracle(ps, dl, 2)
    assert pruned == 1, pruned
    assert torch.equal(topk, torch.tensor(exp_topk, dtype=torch.int64)), topk.tolist()
    # 更大的低分长尾：block_size=2 时中间整块被剪掉
    postings = [
        [[d, 1] for d in range(20)] + [[20, 50]],
        [[0, 1], [20, 50]],
    ]
    ps2 = [_t(p) for p in postings]
    dl2 = torch.tensor([6] * 21, dtype=torch.int64)
    _, topk2, pruned2 = solve(ps2, dl2, k=1, block_size=2)
    _, exp2, _ = _oracle(ps2, dl2, 1)
    assert torch.equal(topk2, torch.tensor(exp2, dtype=torch.int64)), topk2.tolist()
    assert pruned2 > 0, pruned2


def check_skip_and_candidates(solve):
    n = 25
    even = [[d, 1] for d in range(0, n, 2)]
    mid = [[6, 2], [12, 2], [18, 2], [24, 2]]
    few = [[12, 3], [24, 3]]
    ps = [_t(even), _t(mid), _t(few)]
    dl = torch.tensor([4] * n, dtype=torch.int64)
    for bs in (1, 2, 4, 64):
        cand, topk, pruned = solve(ps, dl, k=3, block_size=bs)
        assert torch.equal(cand, torch.tensor([12, 24], dtype=torch.int64)), \
            (bs, cand.tolist())
        _, exp_t, _ = _oracle(ps, dl, 3)
        assert torch.equal(topk, torch.tensor(exp_t, dtype=torch.int64)), \
            (bs, topk.tolist())
    # 有一条为空 -> AND 交集为空
    cand, _, _ = solve([ps[0], _t([]).reshape(0, 2)], dl, k=2, block_size=2)
    assert cand.numel() == 0


def check_empty_query(solve):
    dl = torch.tensor([3, 3, 3], dtype=torch.int64)
    cand, topk, pruned = solve([], dl, k=5)
    assert cand.shape == (0,) and topk.shape == (0,)
    assert cand.dtype == torch.int64 and topk.dtype == torch.int64
    assert pruned == 0


def check_missing_term(solve):
    dl = torch.tensor([4, 4, 4, 4], dtype=torch.int64)
    ps = [_t([[0, 2], [1, 1], [3, 2]]), _t([]).reshape(0, 2),
          _t([[1, 3], [3, 1]])]
    cand, topk, _ = solve(ps, dl, k=2, block_size=2)
    assert cand.numel() == 0, cand.tolist()
    _, exp_t, _ = _oracle(ps, dl, 2)
    assert torch.equal(topk, torch.tensor(exp_t, dtype=torch.int64)), topk.tolist()


def check_ties_ascending(solve):
    dl = torch.tensor([3, 3, 3], dtype=torch.int64)
    ps = [_t([[0, 2], [1, 2], [2, 2]])]
    cand, topk, pruned = solve(ps, dl, k=2, block_size=2)
    assert torch.equal(topk, torch.tensor([0, 1], dtype=torch.int64)), topk.tolist()
    assert pruned > 0, pruned
    _, topk, _ = solve(ps, dl, k=10, block_size=2)
    assert torch.equal(topk, torch.tensor([0, 1, 2], dtype=torch.int64)), topk.tolist()


def check_k_truncation(solve):
    dl = torch.tensor(EXAMPLE_LENS, dtype=torch.int64)
    ps = [_t(p) for p in EXAMPLE_POSTINGS]
    cand, topk, pruned = solve(ps, dl, k=0, block_size=2)
    assert cand.tolist() == EXAMPLE_CANDIDATES
    assert topk.shape == (0,) and pruned == 0
    _, topk, _ = solve(ps, dl, k=100, block_size=2)
    assert torch.equal(topk, torch.tensor([5, 0, 1, 2, 3, 4], dtype=torch.int64)), \
        topk.tolist()
    assert topk.shape == (6,)


def check_single_term(solve):
    dl = torch.tensor([3, 3, 3], dtype=torch.int64)
    ps = [_t([[0, 1], [1, 3], [2, 2]])]
    cand, topk, _ = solve(ps, dl, k=2, block_size=1)
    assert torch.equal(cand, torch.tensor([0, 1, 2], dtype=torch.int64))
    assert torch.equal(topk, torch.tensor([1, 2], dtype=torch.int64)), topk.tolist()


def check_duplicate_terms(solve):
    dl = torch.tensor([4, 4, 4], dtype=torch.int64)
    p = _t([[0, 2], [1, 1], [2, 3]])
    c1, t1, _ = solve([p], dl, k=3, block_size=2)
    c2, t2, _ = solve([p, p], dl, k=3, block_size=2)
    assert torch.equal(c1, c2), (c1.tolist(), c2.tolist())
    assert torch.equal(t1, t2), (t1.tolist(), t2.tolist())
    _, exp_t, _ = _oracle([p, p], dl, 3)
    assert torch.equal(t2, torch.tensor(exp_t, dtype=torch.int64)), t2.tolist()


def check_invalid_inputs(solve):
    dl = torch.tensor([3, 3, 3], dtype=torch.int64)
    good = _t([[0, 1], [1, 2]])

    def expect(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    expect(lambda: solve(good, dl))
    expect(lambda: solve(good, dl.to(torch.float32)))
    expect(lambda: solve([], torch.zeros(0, dtype=torch.int64)))
    expect(lambda: solve([], torch.tensor([3, -1, 3])))
    expect(lambda: solve([_t([[0, 1], [1, 2, 3]])], dl))
    expect(lambda: solve([_t([[1, 1], [1, 2]])], dl))
    expect(lambda: solve([_t([[2, 1], [1, 2]])], dl))
    expect(lambda: solve([_t([[0, 1], [3, 2]])], dl))
    expect(lambda: solve([_t([[0, 0]])], dl))
    expect(lambda: solve([good], dl, k=-1))
    expect(lambda: solve([good], dl, k1=0.0))
    expect(lambda: solve([good], dl, b=-0.1))
    expect(lambda: solve([good], dl, block_size=0))
    expect(lambda: solve({"term": good}, dl))
