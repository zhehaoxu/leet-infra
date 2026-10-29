import math

import torch

# 示例期望值（实测回填）：query=[[1,0],[0,1]]，doc 见下，chunk_size=2
EXAMPLE_QUERY = [[1.0, 0.0], [0.0, 1.0]]
EXAMPLE_DOC = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.6, 0.8]]
EXAMPLE_CHUNK_SIZE = 2
EXAMPLE_BASE = 2.0
EXAMPLE_BASE_IDX = [0, 2]
EXAMPLE_CHUNK_MAXSIM = [
    [1.0, 0.0, 0.6000000238418579],
    [0.0, 1.0, 0.800000011920929],
]
EXAMPLE_MAXPOOLED = 2.0
EXAMPLE_REPEATED = 3.4000000953674316


def _atol(dtype):
    return 1e-12 if dtype == torch.float64 else 1e-5


def _close(got, exp, dtype=None):
    dt = dtype or got.dtype
    return torch.allclose(got.to(dt), torch.as_tensor(exp, dtype=dt),
                          atol=_atol(dt), rtol=0)


def check_example(solve):
    q = torch.tensor(EXAMPLE_QUERY)
    d = torch.tensor(EXAMPLE_DOC)
    base, idx, cm, mp, rp = solve(q, d, EXAMPLE_CHUNK_SIZE)
    assert base.shape == () and base.dtype == torch.float32, (base.shape, base.dtype)
    assert abs(base.item() - EXAMPLE_BASE) < 1e-6, base.item()
    assert torch.equal(idx, torch.tensor(EXAMPLE_BASE_IDX)), idx.tolist()
    assert idx.dtype == torch.int64
    assert cm.shape == (2, 3) and cm.dtype == torch.float32, (cm.shape, cm.dtype)
    assert _close(cm, EXAMPLE_CHUNK_MAXSIM), cm.tolist()
    assert abs(mp.item() - EXAMPLE_MAXPOOLED) < 1e-6, mp.item()
    assert abs(rp.item() - EXAMPLE_REPEATED) < 1e-6, rp.item()


check_example.example = True


def _norm_row(row):
    s = 0.0
    for v in row:
        s += float(v) * float(v)
    n = math.sqrt(s)
    if n == 0.0:
        return [0.0] * len(row)
    return [float(v) / n for v in row]


def _oracle(query, doc, chunk_size):
    """纯 Python 逐 i/j 重算余弦，再逐块循环求 max；结构与矩阵化解法不同。"""
    q = query.tolist()
    d = doc.tolist()
    lq, ld, dim = len(q), len(d), query.shape[1]
    qn = [_norm_row(r) for r in q]
    dn = [_norm_row(r) for r in d]
    cos = []
    for i in range(lq):
        row = []
        for j in range(ld):
            acc = 0.0
            for k in range(dim):
                acc += qn[i][k] * dn[j][k]
            row.append(acc)
        cos.append(row)

    idx = []
    for i in range(lq):
        if ld == 0:
            idx.append(-1)
            continue
        best_j, best_v = 0, cos[i][0]
        for j in range(1, ld):
            if cos[i][j] > best_v:
                best_v, best_j = cos[i][j], j
        idx.append(best_j)

    base = sum(max(cos[i]) for i in range(lq)) if ld > 0 else 0.0

    cols = []
    c = 0
    while c * chunk_size < ld:
        lo = c * chunk_size
        hi = min(lo + chunk_size, ld)
        cols.append([max(cos[i][lo:hi]) for i in range(lq)])
        c += 1
    nc = len(cols)
    mat = [[cols[c][i] for c in range(nc)] for i in range(lq)]
    if nc > 0:
        maxpooled = sum(max(cols[c][i] for c in range(nc)) for i in range(lq))
        repeated = sum(cols[c][i] for c in range(nc) for i in range(lq))
    else:
        maxpooled = 0.0
        repeated = 0.0

    dt = query.dtype
    return (torch.tensor(base, dtype=dt),
            torch.tensor(idx, dtype=torch.int64),
            torch.tensor(mat, dtype=dt).reshape(lq, nc),
            torch.tensor(maxpooled, dtype=dt),
            torch.tensor(repeated, dtype=dt))


def check_matches_oracle(solve):
    cases = [
        (torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
         torch.tensor([[1.0, 0.0], [2.0, 1.0], [0.0, 3.0], [-1.0, 0.0],
                       [1.0, 1.0], [0.5, -2.0]]), 2),
        (torch.tensor([[1.0, 2.0, -1.0], [0.0, 1.0, 1.0], [-2.0, 0.0, 1.0]]),
         torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 2.0],
                       [0.0, -2.0, 1.0], [1.0, 1.0, 1.0], [2.0, -1.0, 0.0],
                       [0.0, 0.0, 0.0]]), 3),
        (torch.tensor([[1.0, 0.0, 0.0, 2.0]]),
         torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0],
                       [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 4.0],
                       [1.0, 1.0, 1.0, 1.0]]), 5),
        (torch.tensor([[1.0, -1.0], [-2.0, 1.0], [0.0, 0.0], [3.0, 1.0]]),
         torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]), 1),
        (torch.zeros(0, 3), torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                          [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]), 2),
        (torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), torch.zeros(0, 3), 2),
        (torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
         torch.tensor([[1.0, 0.0], [1.0, 1.0]]), 10),
    ]
    for k, (q, d, cs) in enumerate(cases):
        got = solve(q, d, cs)
        exp = _oracle(q, d, cs)
        atol = _atol(q.dtype)
        assert got[0].shape == () and abs(got[0].item() - exp[0].item()) <= atol, (k, got[0])
        assert torch.equal(got[1], exp[1]), (k, got[1].tolist(), exp[1].tolist())
        assert got[2].shape == exp[2].shape, (k, got[2].shape, exp[2].shape)
        assert torch.allclose(got[2], exp[2], atol=atol, rtol=0), (k, got[2].tolist())
        assert abs(got[3].item() - exp[3].item()) <= atol, (k, got[3], exp[3])
        assert abs(got[4].item() - exp[4].item()) <= atol, (k, got[4], exp[4])
        assert got[1].dtype == torch.int64 and got[2].dtype == q.dtype


def check_tie_smallest_index(solve):
    # 两行完全相同的文档向量：并列时取最小下标 1
    q = torch.tensor([[1.0, 0.0]])
    d = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    _, idx, cm, _, _ = solve(q, d, 2)
    assert torch.equal(idx, torch.tensor([1])), idx.tolist()
    assert _close(cm, [[1.0, 1.0]]), cm.tolist()
    # 三行全等且余弦为负：仍取最靠前的 0
    _, idx, _, _, _ = solve(q, torch.tensor([[-1.0, 0.0], [-1.0, 0.0]]), 1)
    assert torch.equal(idx, torch.tensor([0])), idx.tolist()
    # 零 query 词：所有余弦都是 0，最大下标按约定为 0
    _, idx, _, _, _ = solve(torch.zeros(1, 2),
                            torch.tensor([[1.0, 0.0], [0.0, 1.0]]), 1)
    assert torch.equal(idx, torch.tensor([0])), idx.tolist()


def check_zero_vectors(solve):
    # query 的第二行是零向量；doc 第一行是零向量：相似度均按 0 处理
    q = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    d = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    base, idx, cm, mp, rp = solve(q, d, 1)
    assert torch.isfinite(cm).all() and torch.isfinite(base)
    assert abs(base.item() - 1.0) < 1e-6, base.item()
    assert torch.equal(idx, torch.tensor([0, 1])), idx.tolist()
    assert _close(cm, [[0.0, 0.0], [0.0, 1.0]]), cm.tolist()
    assert abs(rp.item() - 1.0) < 1e-6 and abs(mp.item() - 1.0) < 1e-6
    # doc 全零：所有 max 为 0
    base, idx, cm, mp, rp = solve(torch.ones(1, 2), torch.zeros(3, 2), 2)
    assert base.item() == 0.0 and torch.equal(idx, torch.tensor([0]))
    assert _close(cm, [[0.0, 0.0]]), cm.tolist()
    assert mp.item() == 0.0 and rp.item() == 0.0


def check_tail_chunk(solve):
    # Ld=5、chunk_size=2 -> 3 块，尾块只有 1 个 token
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    d = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    base, idx, cm, mp, rp = solve(q, d, 2)
    assert cm.shape == (2, 3), cm.shape
    assert _close(cm, [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), cm.tolist()
    assert abs(base.item() - 2.0) < 1e-6 and abs(mp.item() - 2.0) < 1e-6
    assert abs(rp.item() - 3.0) < 1e-6, rp.item()
    assert torch.equal(idx, torch.tensor([0, 2])), idx.tolist()


def check_single_chunk(solve):
    # chunk_size >= Ld：只有一块，repeated 必须等于 base
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    d = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])
    base, _, cm, mp, rp = solve(q, d, 4)
    assert cm.shape == (2, 1), cm.shape
    assert abs(mp.item() - base.item()) < 1e-6
    assert abs(rp.item() - base.item()) < 1e-6
    # chunk_size 大于 Ld 也一样
    base2, _, cm2, mp2, rp2 = solve(q, d, 99)
    assert cm2.shape == (2, 1)
    assert abs(mp2.item() - base2.item()) < 1e-6
    assert abs(rp2.item() - base2.item()) < 1e-6


def check_repeated_overcount(solve):
    # 同一个 query 词在 4 个单 token 块里命中两次：重复计分翻倍
    q = torch.tensor([[1.0, 0.0]])
    d = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    base, _, cm, mp, rp = solve(q, d, 1)
    assert _close(cm, [[1.0, 0.0, 1.0, 0.0]]), cm.tolist()
    assert abs(base.item() - 1.0) < 1e-6
    assert abs(mp.item() - 1.0) < 1e-6
    assert abs(rp.item() - 2.0) < 1e-6, rp.item()
    assert abs(rp.item() - mp.item() - 1.0) < 1e-6


def check_empty_inputs(solve):
    # 空文档：C=0，base=0，下标全 -1
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    base, idx, cm, mp, rp = solve(q, torch.zeros(0, 2), 3)
    assert base.shape == () and base.item() == 0.0
    assert torch.equal(idx, torch.tensor([-1, -1])), idx.tolist()
    assert cm.shape == (2, 0) and cm.dtype == torch.float32
    assert mp.item() == 0.0 and rp.item() == 0.0
    # 空 query：块矩阵第一维为 0
    d = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    base, idx, cm, mp, rp = solve(torch.zeros(0, 2), d, 2)
    assert base.item() == 0.0 and idx.shape == (0,)
    assert cm.shape == (0, 2), cm.shape
    assert mp.item() == 0.0 and rp.item() == 0.0
    # 双空
    base, idx, cm, _, _ = solve(torch.zeros(0, 2), torch.zeros(0, 2), 2)
    assert base.item() == 0.0 and idx.shape == (0,) and cm.shape == (0, 0)


def check_dtype_float64(solve):
    q = torch.tensor([[3.0, 4.0], [0.0, 2.0]], dtype=torch.float64)
    d = torch.tensor([[1.0, 0.0], [0.0, 5.0], [-3.0, 4.0]], dtype=torch.float64)
    got = solve(q, d, 2)
    exp = _oracle(q, d, 2)
    for g, e in zip(got[:1], exp[:1]):
        assert g.dtype == torch.float64 and e.dtype == torch.float64
    assert got[0].dtype == torch.float64
    assert got[2].dtype == torch.float64
    assert got[1].dtype == torch.int64
    assert abs(got[0].item() - exp[0].item()) <= 1e-12, (got[0], exp[0])
    assert torch.equal(got[1], exp[1]), (got[1], exp[1])
    assert torch.allclose(got[2], exp[2], atol=1e-12, rtol=0), (got[2], exp[2])
    assert abs(got[3].item() - exp[3].item()) <= 1e-12
    assert abs(got[4].item() - exp[4].item()) <= 1e-12


def _expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def check_errors(solve):
    q = torch.zeros(1, 2)
    d = torch.zeros(2, 2)
    _expect_value_error(lambda: solve(torch.zeros(1, 2, 2), d, 1))
    _expect_value_error(lambda: solve(q, torch.zeros(2), 1))
    _expect_value_error(lambda: solve(q, torch.zeros(2, 3), 1))
    _expect_value_error(lambda: solve(q, d.double(), 1))
    _expect_value_error(lambda: solve(torch.zeros(1, 2, dtype=torch.long),
                                      torch.zeros(2, 2, dtype=torch.long), 1))
    _expect_value_error(lambda: solve(q, d, 0))
    _expect_value_error(lambda: solve(q, d, -2))
