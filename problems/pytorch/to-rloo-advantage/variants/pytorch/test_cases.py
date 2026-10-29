import torch

REWARDS = torch.tensor([[1.0, 0.0, 0.5], [2.0, 2.0, 0.0]], dtype=torch.float64)
EXAMPLE = [[0.75, -0.75, 0.0], [1.0, 1.0, -2.0]]


def _oracle(rewards):
    out = []
    for row in rewards:
        g = len(row)
        total = sum(row)
        out.append([r - (total - r) / (g - 1) for r in row])
    return out


def check_example(solve):
    y = solve(REWARDS)
    assert y.shape == (2, 3) and y.dtype == torch.float64, (y.shape, y.dtype)
    assert y.tolist() == EXAMPLE, y.tolist()


check_example.example = True


def check_oracle(solve):
    torch.manual_seed(29)
    for _ in range(6):
        b, g = 4, 6
        r = torch.randn(b, g, dtype=torch.float64)
        got = solve(r).tolist()
        exp = _oracle(r.tolist())
        for bi in range(b):
            for gi in range(g):
                assert abs(got[bi][gi] - exp[bi][gi]) < 1e-12, (bi, gi, got[bi][gi], exp[bi][gi])


def check_zero_sum(solve):
    torch.manual_seed(31)
    y = solve(torch.randn(5, 4, dtype=torch.float64))
    assert torch.allclose(y.sum(dim=1), torch.zeros(5, dtype=torch.float64), atol=1e-12), y.sum(dim=1)


def check_g2_raw_difference(solve):
    # G=2 时留一均值就是另一个样本本身：A = [r0-r1, r1-r0]
    y = solve(torch.tensor([[1.0, 0.0]], dtype=torch.float64))
    assert y.tolist() == [[1.0, -1.0]], y.tolist()
    # 若误用分母 G（而不是 G-1），G=3 的 [1,0,3] 会得到 [-1/3, -4/3, 5/3]
    y2 = solve(torch.tensor([[1.0, 0.0, 3.0]], dtype=torch.float64))
    assert y2.tolist() == [[-0.5, -2.0, 2.5]], y2.tolist()


def check_uniform_rewards_zero(solve):
    y = solve(torch.full((3, 5), 2.5, dtype=torch.float64))
    assert torch.equal(y, torch.zeros(3, 5, dtype=torch.float64)), y


def check_float32_input(solve):
    y = solve(REWARDS.to(torch.float32))
    assert y.dtype == torch.float64
    assert torch.allclose(y, torch.tensor(EXAMPLE, dtype=torch.float64), atol=1e-6)


def check_errors(solve):
    def bad(fn, desc):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError: {desc}")

    bad(lambda: solve([[1.0, 0.0]]), "非张量")
    bad(lambda: solve(torch.tensor([1.0, 0.0])), "非二维")
    bad(lambda: solve(torch.tensor([[1, 0]], dtype=torch.int64)), "非浮点")
    bad(lambda: solve(torch.tensor([[1.0]], dtype=torch.float64)), "G=1")
    bad(lambda: solve(torch.zeros(0, 3, dtype=torch.float64)), "B=0")
    bad(lambda: solve(torch.tensor([[float("nan"), 0.0]], dtype=torch.float64)), "含 NaN")
