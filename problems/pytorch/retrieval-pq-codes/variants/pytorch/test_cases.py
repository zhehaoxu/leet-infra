import torch


def _oracle(x, codebook):
    n = x.shape[0]
    m, ksub, d = codebook.shape
    out = []
    for i in range(n):
        row = []
        for j in range(m):
            best, bi = None, 0
            for c in range(ksub):
                s = 0.0
                for t in range(d):
                    v = float(x[i, j * d + t]) - float(codebook[j, c, t])
                    s += v * v
                if best is None or s < best:
                    best, bi = s, c
            row.append(bi)
        out.append(row)
    return torch.tensor(out, dtype=torch.long)


def check_example(solve):
    cb = torch.tensor([[[0.0, 0.0], [10.0, 10.0]],
                       [[0.0, 0.0], [0.0, 5.0]]])
    x = torch.tensor([[0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 0.0, 6.0]])
    y = solve(x, cb)
    assert y.tolist() == [[0, 0], [0, 1]], f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    g = torch.Generator().manual_seed(9)
    cb = torch.randn(3, 4, 2, generator=g)          # m=3, ksub=4, d=2 -> D=6
    x = torch.randn(6, 6, generator=g)
    y = solve(x, cb)
    assert torch.equal(y, _oracle(x, cb)), (y.tolist(), _oracle(x, cb).tolist())


def check_shape_dtype(solve):
    g = torch.Generator().manual_seed(21)
    cb = torch.randn(3, 5, 2, generator=g)          # m=3
    x = torch.randn(7, 6, generator=g)
    y = solve(x, cb)
    assert y.shape == (7, 3) and y.dtype == torch.int64, (y.shape, y.dtype)


def check_code_range(solve):
    g = torch.Generator().manual_seed(23)
    cb = torch.randn(4, 6, 3, generator=g)          # ksub=6
    x = torch.randn(10, 12, generator=g)
    y = solve(x, cb)
    assert bool((y >= 0).all()) and bool((y < 6).all()), y.tolist()


def check_exact_reconstruction(solve):
    # 由已知子码拼回向量（各段恰为码字），再编码应原样还原子码
    cb = torch.tensor([[[0.0, 0.0], [5.0, 5.0], [9.0, 0.0]],
                       [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    target = [[2, 0], [1, 1]]
    x = torch.tensor([
        cb[0][2].tolist() + cb[1][0].tolist(),
        cb[0][1].tolist() + cb[1][1].tolist(),
    ], dtype=torch.float32)
    y = solve(x, cb)
    assert y.tolist() == target, y.tolist()
