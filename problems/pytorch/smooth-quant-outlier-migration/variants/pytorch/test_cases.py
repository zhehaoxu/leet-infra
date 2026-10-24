import torch


def _oracle(X, W, alpha):
    """独立纯 Python：逐通道 absmax → 因子 → 迁移，全手工循环。"""
    T, C = X.shape
    O = W.shape[0]
    s = []
    for j in range(C):
        ax = max(abs(float(X[i][j])) for i in range(T))
        aw = max(abs(float(W[o][j])) for o in range(O))
        if aw > 0:
            v = (ax ** alpha) / (aw ** (1 - alpha))
        else:
            v = ax ** alpha
        if v == 0:
            v = 1.0
        s.append(v)
    Xhat = [[float(X[i][j]) / s[j] for j in range(C)] for i in range(T)]
    What = [[float(W[o][j]) * s[j] for j in range(C)] for o in range(O)]
    return s, Xhat, What


def check_example(solve):
    X = torch.tensor([[1.0, 2.0, 100.0], [3.0, 0.5, 4.0], [-2.0, 1.0, 8.0]])
    W = torch.tensor([[0.1, 5.0, 0.2], [1.0, 0.3, 6.0]])
    s, Xh, Wh = solve(X, W, 0.5)
    assert torch.allclose(s, torch.tensor([1.732051, 0.632456, 4.082483]), atol=1e-4), s
    os, oXh, oWh = _oracle(X, W, 0.5)
    assert torch.allclose(Xh, torch.tensor(oXh), atol=1e-4), Xh
    assert torch.allclose(Wh, torch.tensor(oWh), atol=1e-4), Wh


def check_matches_oracle(solve):
    torch.manual_seed(5)
    for _ in range(30):
        T = int(torch.randint(1, 6, (1,)))
        O = int(torch.randint(1, 6, (1,)))
        C = int(torch.randint(1, 8, (1,)))
        X = torch.randn(T, C) * 10
        W = torch.randn(O, C)
        alpha = float(torch.rand(1))
        s, Xh, Wh = solve(X, W, alpha)
        os, oXh, oWh = _oracle(X, W, alpha)
        assert torch.allclose(s, torch.tensor(os), atol=1e-4), (s, os)
        assert torch.allclose(Xh, torch.tensor(oXh), atol=1e-3), Xh
        assert torch.allclose(Wh, torch.tensor(oWh), atol=1e-3), Wh


def check_invariance(solve):
    # 迁移不改变 X W^T
    X = torch.randn(4, 5)
    W = torch.randn(3, 5)
    s, Xh, Wh = solve(X, W, 0.5)
    assert torch.allclose(Xh @ Wh.t(), X @ W.t(), atol=1e-3), (Xh @ Wh.t(), X @ W.t())


def check_alpha_zero_one(solve):
    X = torch.tensor([[2.0, 8.0]])
    W = torch.tensor([[4.0, 1.0]])
    s0, _, _ = solve(X, W, 0.0)   # s = aw^1 分量在分母 → alpha=0: ax^0/aw^1 = 1/aw
    s1, _, _ = solve(X, W, 1.0)   # alpha=1: ax/aw^0 = ax
    assert torch.allclose(s0, 1.0 / W.abs().max(dim=0).values, atol=1e-5), s0
    assert torch.allclose(s1, X.abs().max(dim=0).values, atol=1e-5), s1


check_example.example = True
