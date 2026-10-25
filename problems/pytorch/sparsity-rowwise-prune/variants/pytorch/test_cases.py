import torch

W = torch.tensor([[1.0, -5.0, 3.0, -2.0], [0.5, 9.0, -2.0, -8.0]])


def _oracle(w, s):
    r, c = w.shape
    k = int(round(s * c))
    out = torch.zeros_like(w)
    for i in range(r):
        row = w[i].tolist()
        order = sorted(range(c), key=lambda j: (abs(row[j]), j))
        keep = set(order[k:])
        for j in range(c):
            out[i, j] = row[j] if j in keep else 0.0
    return out


def check_example(solve):
    y = solve(W, 0.5)
    exp = torch.tensor([[0.0, -5.0, 3.0, 0.0], [0.0, 9.0, 0.0, -8.0]])
    assert torch.allclose(y, exp, atol=1e-5, rtol=1e-5), f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(9)
    for shape in [(4, 8), (5, 7), (2, 10)]:
        w = torch.randn(*shape) * 3
        for s in [0.0, 0.25, 0.5, 0.7]:
            assert torch.allclose(solve(w, s), _oracle(w, s), atol=1e-5, rtol=1e-5)


def check_each_row_same_keep(solve):
    torch.manual_seed(2)
    w = torch.randn(5, 12)
    s = 0.5
    k = int(round(s * 12))
    y = solve(w, s)
    nz = (y != 0).sum(dim=1)
    assert torch.all(nz == (12 - k)), nz.tolist()


def check_local_differs_from_global(solve):
    # 行1 整体幅值都很小：局部剪枝每行都剪 k 个；全局剪枝可能把行1 全剪、行0 几乎不动 → 结果不同
    w = torch.tensor([[100.0, 200.0, 300.0, 400.0], [0.001, 0.002, 0.003, 0.004]])
    y = solve(w, 0.5)
    # 局部：每行各剪 2 个 → 行0 仍有 2 个非零、行1 仍有 2 个非零
    assert (y[0] != 0).sum().item() == 2
    assert (y[1] != 0).sum().item() == 2


def check_bounds(solve):
    assert torch.equal(solve(W, 0.0), W)
    assert torch.equal(solve(W, 1.0), torch.zeros_like(W))
