import torch


def _cos(a, b):
    na = sum(float(v) * float(v) for v in a) ** 0.5
    nb = sum(float(v) * float(v) for v in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return sum(float(a[t]) * float(b[t]) for t in range(len(a))) / (na * nb)


def _oracle(query, docs, lam, k):
    n = docs.shape[0]
    rel = [_cos([docs[i, t] for t in range(docs.shape[1])], query) for i in range(n)]
    selected = []
    remaining = list(range(n))
    for _ in range(k):
        best_i, best_v = None, None
        for i in remaining:
            pen = 0.0
            for s in selected:
                c = _cos([docs[i, t] for t in range(docs.shape[1])],
                         [docs[s, t] for t in range(docs.shape[1])])
                pen = max(pen, c)
            v = lam * rel[i] - (1 - lam) * pen
            if best_v is None or v > best_v + 1e-12:
                best_v, best_i = v, i
        selected.append(best_i)
        remaining.remove(best_i)
    return torch.tensor(selected, dtype=torch.long)


def check_example(solve):
    q = torch.tensor([1.0, 1.0, 0.0])
    docs = torch.tensor([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]])
    y = solve(q, docs, 0.5, 3)
    assert y.tolist() == [1, 2, 0], f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    cases = [
        (torch.tensor([1.0, 1.0, 0.0]),
         torch.tensor([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]]), 0.5, 3),
        (torch.tensor([1.0, 1.0, 0.0]),
         torch.tensor([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]]), 1.0, 3),
        (torch.tensor([1.0, 1.0, 1.0]),
         torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.1, 1.0], [1.0, 1.0, 0.0]]), 0.5, 4),
        (torch.tensor([1.0, 1.0, 0.0]),
         torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 0.5, 3),  # 零范数
        (torch.tensor([0.3, 1.0, 0.7]),
         torch.tensor([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.2],
                       [0.9, 0.4, 0.1], [0.2, 0.9, 0.3]]), 0.4, 5),
    ]
    for q, docs, lam, k in cases:
        y = solve(q, docs, lam, k)
        assert torch.equal(y, _oracle(q, docs, lam, k)), (lam, y.tolist())


def check_pure_relevance_lam1(solve):
    # λ=1 退化为纯相关性 top-k：q=[1,0,0], rel(d0=.707, d1=1, d2=0) -> [1,0,2]
    q = torch.tensor([1.0, 0.0, 0.0])
    docs = torch.tensor([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y = solve(q, docs, 1.0, 3)
    assert y.tolist() == [1, 0, 2], y.tolist()


def check_pure_diversity_lam0(solve):
    # λ=0：首选并列取小下标=0，之后纯取最多样 -> [0,2,1]
    q = torch.tensor([1.0, 0.0, 0.0])
    docs = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.05, 0.0], [0.0, 1.0, 0.0]])
    y = solve(q, docs, 0.0, 3)
    assert y.tolist() == [0, 2, 1], y.tolist()


def check_length_distinct_dtype(solve):
    q = torch.tensor([1.0, 1.0, 1.0])
    docs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                         [1.0, 1.0, 0.0], [0.5, 0.5, 0.5], [1.0, 0.2, 0.7]])
    y = solve(q, docs, 0.5, 4)
    assert y.dtype == torch.int64 and y.shape == (4,), (y.dtype, y.shape)
    assert len(set(y.tolist())) == 4 and all(0 <= v < 6 for v in y.tolist()), y.tolist()
