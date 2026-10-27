import torch

W0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])               # (2,2)
LAMBDA_B = torch.tensor([1.0, -1.0])                      # (2,)
LAMBDA_D = torch.tensor([2.0])                            # (1,) r=1
A = torch.tensor([[1.0, 1.0]])                            # (1,2)
B = torch.tensor([[1.0], [-1.0]])                         # (2,1)
X = torch.tensor([[1.0, 2.0]])                            # (1,2)


def _oracle(W0, lambda_b, lambda_d, A, B, x):
    """独立重算：显式 ΔW 矩阵式 + 逐标量点积（结构异于参考解的函数式广播路径）。"""
    d_out, d_in = W0.shape
    r = lambda_d.shape[0]
    lb = [float(v) for v in lambda_b]
    ld = [float(v) for v in lambda_d]
    Am = [[float(A[t, j]) for j in range(d_in)] for t in range(r)]
    Bm = [[float(B[i, t]) for t in range(r)] for i in range(d_out)]
    dW = [[sum(lb[i] * Bm[i][t] * ld[t] * Am[t][j] for t in range(r))
           for j in range(d_in)] for i in range(d_out)]
    y = []
    for s in range(x.shape[0]):
        row = []
        for i in range(d_out):
            acc = sum(float(x[s, j]) * (float(W0[i, j]) + dW[i][j]) for j in range(d_in))
            row.append(acc)
        y.append(row)
    return torch.tensor(y, dtype=torch.float64).to(torch.float32), d_out + r


def check_example(solve):
    y, n = solve(W0, LAMBDA_B, LAMBDA_D, A, B, X)
    assert torch.allclose(y, torch.tensor([[7.0, 8.0]]), atol=1e-5, rtol=1e-5), (
        f"got {y.tolist()}")
    assert n == 3, f"n_trainable 应为 3，got {n}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(131)
    d_in, d_out, r, n_sam = 5, 4, 3, 2
    w0 = torch.randn(d_out, d_in)
    lb, ld = torch.randn(d_out), torch.randn(r)
    a, b = torch.randn(r, d_in), torch.randn(d_out, r)
    x = torch.randn(n_sam, d_in)
    y, n = solve(w0, lb, ld, a, b, x)
    exp_y, exp_n = _oracle(w0, lb, ld, a, b, x)
    assert torch.allclose(y, exp_y, atol=1e-5, rtol=1e-5), (
        "函数式前向与显式 ΔW 重建不一致")
    assert n == exp_n, f"n_trainable 应为 {exp_n}，got {n}"


def check_zero_lambda_b_identity(solve):
    """λb=0（PEFT 初始化）→ y == x@W0^T。"""
    torch.manual_seed(141)
    d_in, d_out, r = 4, 3, 2
    w0 = torch.randn(d_out, d_in)
    ld = torch.full((r,), 0.1)
    a, b = torch.randn(r, d_in), torch.randn(d_out, r)
    x = torch.randn(2, d_in)
    y, n = solve(w0, torch.zeros(d_out), ld, a, b, x)
    base = (x.to(torch.float64) @ w0.to(torch.float64).T).to(torch.float32)
    assert torch.allclose(y, base, atol=1e-6, rtol=1e-6), "λb=0 时应等于纯基座输出"
    assert n == d_out + r, f"n_trainable 应为 {d_out+r}，got {n}"


def check_param_account(solve):
    torch.manual_seed(151)
    for d_in, d_out, r in ((6, 4, 2), (3, 7, 5)):
        w0 = torch.randn(d_out, d_in)
        lb, ld = torch.randn(d_out), torch.randn(r)
        a, b = torch.randn(r, d_in), torch.randn(d_out, r)
        x = torch.randn(1, d_in)
        _, n = solve(w0, lb, ld, a, b, x)
        assert n == d_out + r, f"n_trainable 应为 {d_out+r}，got {n}"


def check_lambda_b_per_output(solve):
    """λb 某位翻倍：只有该输出列变化，且变化量 = 该列原适配量。"""
    torch.manual_seed(161)
    d_in, d_out, r = 4, 3, 2
    w0 = torch.randn(d_out, d_in)
    lb = torch.randn(d_out)
    ld = torch.randn(r)
    a, b = torch.randn(r, d_in), torch.randn(d_out, r)
    x = torch.randn(2, d_in)
    y0, _ = solve(w0, lb, ld, a, b, x)
    lb2 = lb.clone()
    lb2[1] = 2 * lb[1]
    y1, _ = solve(w0, lb2, ld, a, b, x)
    base = (x.to(torch.float64) @ w0.to(torch.float64).T).to(torch.float32)
    for i in range(d_out):
        if i == 1:
            assert torch.allclose(y1[:, i] - y0[:, i], y0[:, i] - base[:, i],
                                  atol=1e-5, rtol=1e-5), "翻倍列的增量应等于原适配量"
        else:
            assert torch.allclose(y1[:, i], y0[:, i], atol=1e-6, rtol=1e-6), (
                f"输出列 {i} 不应受其他位 λb 影响")
