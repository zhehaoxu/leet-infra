import torch


def _rtn(x, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = max(abs(v) for row in x for v in row) or 1.0
    return [[max(-qmax - 1, min(qmax, round(v / scale * qmax))) / qmax * scale for v in row] for row in x]


def _wdq(W, x_bar, a, bits):
    O, C = W.shape
    raw = [float(x_bar[j]) ** a for j in range(C)]
    m = sum(raw) / len(raw)
    s = [r / m for r in raw]
    Ws = [[float(W[o][j]) * s[j] for j in range(C)] for o in range(O)]
    q = _rtn(Ws, bits)
    return [[q[o][j] / s[j] for j in range(C)] for o in range(O)]


def _oracle(W, x_bar, alphas, bits):
    O, C = W.shape
    best, best_err = None, float("inf")
    for a in alphas:
        d = _wdq(W, x_bar, a, bits)
        err = sum((float(W[o][j]) - d[o][j]) ** 2 * float(x_bar[j]) ** 2 for o in range(O) for j in range(C))
        if err < best_err - 1e-12:
            best_err, best = err, a
    return float(best), _wdq(W, x_bar, best, bits)


def check_example(solve):
    W = torch.tensor([[2.0, 1.0, 0.10, 3.0], [1.0, 2.0, 0.12, 2.0], [0.5, 1.0, 0.08, 1.0]])
    x_bar = torch.tensor([0.2, 0.3, 9.0, 0.1])
    a, Wh = solve(W, x_bar, [0.0, 0.25, 0.5, 0.75, 1.0], 4)
    assert a == 1.0, a
    exp = torch.tensor([[2.3143, 1.0286, 0.1029, 3.0857],
                        [0.7714, 2.0571, 0.1200, 1.5429],
                        [0.7714, 1.0286, 0.0857, 1.5429]])
    assert torch.allclose(Wh, exp, atol=1e-3), Wh


def check_matches_oracle(solve):
    torch.manual_seed(6)
    for _ in range(30):
        O = int(torch.randint(1, 5, (1,)))
        C = int(torch.randint(2, 7, (1,)))
        W = torch.randn(O, C)
        x_bar = 0.1 + torch.rand(C) * 3
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        bits = int(torch.randint(2, 9, (1,)))
        a, Wh = solve(W, x_bar, alphas, bits)
        oa, oW = _oracle(W, x_bar, alphas, bits)
        assert a == oa, (a, oa)
        assert torch.allclose(Wh, torch.tensor(oW), atol=1e-4), Wh


def check_alpha_uses_importance(solve):
    # 一个大激活小权重通道：非零 alpha 应优于 alpha=0（保护重要通道）
    W = torch.tensor([[2.0, 0.1], [1.0, 0.12]])
    x_bar = torch.tensor([0.1, 5.0])
    a, _ = solve(W, x_bar, [0.0, 1.0], 2)
    assert a == 1.0, a


def check_uniform_xbar_neutral(solve):
    # x_bar 全相等 → s 恒为 1，任意 alpha 结果相同，选先出现的 0.0
    W = torch.tensor([[0.3, -0.7, 0.5]])
    x_bar = torch.tensor([2.0, 2.0, 2.0])
    a, Wh = solve(W, x_bar, [0.0, 0.5, 1.0], 4)
    assert a == 0.0, a
    _, plain = solve(W, x_bar, [0.0], 4)          # alpha=0 的 qdq
    assert torch.allclose(Wh, plain, atol=1e-5), (Wh, plain)


check_example.example = True
