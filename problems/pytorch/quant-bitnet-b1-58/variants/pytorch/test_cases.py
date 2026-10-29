import torch

W_EX = torch.tensor([[0.5, -0.4, 0.0], [0.2, 0.8, -0.3]])
X_EX = torch.tensor([[1.0, 2.0, 3.0]])


def _oracle(W, X):
    """逐元素重算：absmean、三值化、加减计数 GEMM。"""
    d_out, d_in = W.shape
    b = X.shape[0]
    s = float(W.abs().mean())
    if s == 0:
        T = torch.zeros_like(W, dtype=torch.int64)
        W_hat = torch.zeros_like(W)
        Y = torch.zeros(b, d_out, dtype=W.dtype)
        return s, T, W_hat, Y
    T = torch.clamp(torch.round(W / s), -1, 1).to(torch.int64)
    W_hat = T.to(W.dtype) * s
    Y = torch.zeros(b, d_out, dtype=torch.float64)
    Xd = X.to(torch.float64)
    for i in range(b):
        for o in range(d_out):
            acc = 0.0
            for k in range(d_in):
                acc += float(Xd[i, k]) * int(T[o, k])
            Y[i, o] = s * acc
    return s, T, W_hat, Y.to(W.dtype)


def check_example(solve):
    s, T, W_hat, Y = solve(W_EX, X_EX)
    assert abs(s - 0.36666667461395264) < 1e-6, f"s got {s}"
    exp_T = torch.tensor([[1, -1, 0], [1, 1, -1]], dtype=torch.int64)
    assert torch.equal(T, exp_T), f"T got {T.tolist()}"
    exp_Y = torch.tensor([[-0.36666667461395264, 0.0]])
    assert torch.allclose(Y, exp_Y, atol=1e-5, rtol=1e-5), f"Y got {Y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(11)
    W = torch.randn(5, 7)
    X = torch.randn(3, 7)
    out = solve(W, X)
    exp = _oracle(W, X)
    for a, b, name in zip(out, exp, ["s", "T", "W_hat", "Y"]):
        if name == "s":
            assert abs(float(a) - float(b)) < 1e-6, f"{name} mismatch"
        else:
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"{name} mismatch"


def check_zero_weights(solve):
    W = torch.zeros(2, 3)
    X = torch.ones(1, 3)
    s, T, W_hat, Y = solve(W, X)
    assert s == 0.0
    assert torch.equal(T, torch.zeros(2, 3, dtype=torch.int64))
    assert torch.equal(W_hat, torch.zeros_like(W))
    assert torch.equal(Y, torch.zeros(1, 2))


def check_add_subtract_equivalence(solve):
    """三值 GEMM 结果应等价于 X @ W_hat.T。"""
    torch.manual_seed(7)
    W = torch.randn(4, 6)
    X = torch.randn(2, 6)
    _, _, W_hat, Y = solve(W, X)
    exp = X @ W_hat.T
    assert torch.allclose(Y, exp, atol=1e-5, rtol=1e-5), "GEMM 结果与直接矩阵乘不等价"


def check_scaling_linearity(solve):
    """W 整体缩放 k 倍后，s 与 W_hat 同倍缩放，T 不变，Y 同倍缩放。"""
    torch.manual_seed(3)
    W = torch.randn(3, 5)
    X = torch.randn(2, 5)
    k = 2.5
    s1, T1, _, Y1 = solve(W, X)
    s2, T2, _, Y2 = solve(k * W, X)
    assert torch.equal(T1, T2), "整体缩放不应改变三值码"
    assert abs(s2 - k * s1) < 1e-5, "s 应线性缩放"
    assert torch.allclose(Y2, k * Y1, atol=1e-5, rtol=1e-5), "Y 应线性缩放"
