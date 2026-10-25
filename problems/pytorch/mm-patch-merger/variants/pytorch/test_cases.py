import torch


def _oracle(emb, gh, gw):
    d = emb.shape[1]
    out = torch.zeros((gh // 2) * (gw // 2), 4 * d)
    for i in range(gh // 2):
        for j in range(gw // 2):
            vec = []
            for (r, c) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                vec.extend(emb[(2 * i + r) * gw + (2 * j + c)].tolist())
            out[i * (gw // 2) + j] = torch.tensor(vec)
    return out


def check_example(solve):
    emb = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    y = solve(emb, 2, 2)
    exp = torch.tensor([[0.0, 1, 2, 3, 4, 5, 6, 7]])
    assert torch.allclose(y, exp, atol=1e-5), f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(61)
    for gh, gw, d in [(2, 4, 3), (4, 4, 2), (6, 2, 5)]:
        emb = torch.randn(gh * gw, d)
        assert torch.allclose(solve(emb, gh, gw), _oracle(emb, gh, gw), atol=1e-5)


def check_shape(solve):
    emb = torch.randn(4 * 6, 3)
    y = solve(emb, 4, 6)
    assert y.shape == ((4 // 2) * (6 // 2), 4 * 3), y.shape


def check_bijection(solve):
    torch.manual_seed(62)
    emb = torch.randn(4 * 4, 3)
    y = solve(emb, 4, 4)
    assert torch.equal(torch.sort(y.flatten()).values, torch.sort(emb.flatten()).values)


def check_first_block_order(solve):
    # grid 2x2 且 D=1，值=网格行主序索引 → 首块应为 [0,1,2,3]（左上,右上,左下,右下）
    emb = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    y = solve(emb, 2, 2)
    assert y.flatten().tolist() == [0.0, 1.0, 2.0, 3.0], y.flatten().tolist()
