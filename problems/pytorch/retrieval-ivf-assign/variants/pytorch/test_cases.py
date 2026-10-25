import torch


def _oracle(x, centroids):
    n, D = x.shape
    k = centroids.shape[0]
    out = []
    for i in range(n):
        best, bi = None, 0
        for j in range(k):
            s = 0.0
            for t in range(D):
                v = float(x[i, t]) - float(centroids[j, t])
                s += v * v
            if best is None or s < best:
                best, bi = s, j
        out.append(bi)
    return torch.tensor(out, dtype=torch.long)


def check_example(solve):
    x = torch.tensor([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
    c = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
    y = solve(x, c)
    assert y.tolist() == [0, 1, 0], f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    g = torch.Generator().manual_seed(7)
    x = torch.randn(9, 4, generator=g)
    c = torch.randn(3, 4, generator=g)
    y = solve(x, c)
    assert torch.equal(y, _oracle(x, c)), (y.tolist(), _oracle(x, c).tolist())


def check_single_centroid(solve):
    g = torch.Generator().manual_seed(11)
    x = torch.randn(6, 5, generator=g)
    c = torch.randn(1, 5, generator=g)
    y = solve(x, c)
    assert y.tolist() == [0, 0, 0, 0, 0, 0], y.tolist()


def check_exact_match_hit(solve):
    # 每个向量恰为某个中心 → 应命中该中心本身（距离 0）
    c = torch.tensor([[0.0, 0.0], [1.0, 2.0], [-3.0, 4.0]])
    y = solve(c.clone(), c)
    assert y.tolist() == [0, 1, 2], y.tolist()


def check_range_and_dtype(solve):
    g = torch.Generator().manual_seed(13)
    x = torch.randn(12, 3, generator=g)
    c = torch.randn(4, 3, generator=g)
    y = solve(x, c)
    assert y.dtype == torch.int64, y.dtype
    assert y.shape == (12,), y.shape
    assert bool((y >= 0).all()) and bool((y < 4).all()), y.tolist()
