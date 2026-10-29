import torch


def _quantize(W, scale, qmax):
    q = torch.clamp(torch.round(W / scale * qmax), -qmax - 1, qmax)
    return q / qmax * scale


def _gptq(W, H, bits, damp):
    O, C = W.shape
    qmax = 2 ** (bits - 1) - 1
    scale = float(W.abs().max()) or 1.0
    Hinv = torch.linalg.inv(H + damp * torch.eye(C, dtype=H.dtype, device=H.device))
    Wq = W.clone()
    out = W.clone()
    for i in range(C):
        col = Wq[:, i]
        dq = _quantize(col, scale, qmax)
        err = col - dq
        out[:, i] = dq
        d = Hinv[i, i]
        Wq[:, i + 1:] -= (err / d).unsqueeze(1) * Hinv[i, i + 1:].unsqueeze(0)
    return out


def _oracle(W, H, X, bits, damp, group_size):
    qmax = 2 ** (bits - 1) - 1
    scale = float(W.abs().max()) or 1.0
    W_rtn = _quantize(W, scale, qmax)
    W_gptq = _gptq(W, H, bits, damp)
    Y_true = X @ W.T
    Y_rtn = X @ W_rtn.T
    Y_gptq = X @ W_gptq.T
    mse_rtn = float(((Y_true - Y_rtn) ** 2).mean())
    mse_gptq = float(((Y_true - Y_gptq) ** 2).mean())

    def group_mae(a, b):
        O, C = a.shape
        return (a - b).abs().view(O, C // group_size, group_size).mean(dim=2).flatten()

    probs = torch.tensor([0.25, 0.5, 0.75, 0.9], dtype=W.dtype, device=W.device)
    rtn_q = torch.quantile(group_mae(W, W_rtn), probs).to(W.dtype)
    gptq_q = torch.quantile(group_mae(W, W_gptq), probs).to(W.dtype)
    return (torch.tensor(mse_rtn, dtype=W.dtype),
            torch.tensor(mse_gptq, dtype=W.dtype),
            rtn_q, gptq_q)


def check_example(solve):
    W = torch.tensor([[0.5, -0.4, 0.3, -0.2],
                      [0.2, 0.8, -0.1, 0.6]], dtype=torch.float32)
    H = torch.eye(4, dtype=torch.float32) * 0.1
    X = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                      [0.5, -0.5, 1.0, -1.0]], dtype=torch.float32)
    mse_rtn, mse_gptq, rtn_q, gptq_q = solve(W, H, X, 4, 0.01, 2)
    exp = _oracle(W, H, X, 4, 0.01, 2)
    assert abs(float(mse_rtn) - float(exp[0])) < 1e-5
    assert abs(float(mse_gptq) - float(exp[1])) < 1e-5
    assert torch.allclose(rtn_q, exp[2], atol=1e-5, rtol=1e-5)
    assert torch.allclose(gptq_q, exp[3], atol=1e-5, rtol=1e-5)


check_example.example = True


def check_gptq_improves_mse(solve):
    """当 Hessian 取 X^T X 时，GPTQ 的输出 MSE 应严格小于 RTN。"""
    torch.manual_seed(53)
    C = 64
    W = torch.randn(8, C)
    X = torch.randn(16, C)
    H = X.T @ X / X.shape[0] + 0.001 * torch.eye(C)
    mse_rtn, mse_gptq, _, _ = solve(W, H, X, 4, 0.01, 16)
    assert mse_gptq < mse_rtn, f"GPTQ MSE {mse_gptq} 应小于 RTN MSE {mse_rtn}"


def check_matches_oracle(solve):
    torch.manual_seed(59)
    W = torch.randn(5, 32)
    A = torch.randn(32, 8)
    H = A @ A.T / 8.0 + 0.005 * torch.eye(32)
    X = torch.randn(3, 32)
    out = solve(W, H, X, 4, 0.01, 8)
    exp = _oracle(W, H, X, 4, 0.01, 8)
    for a, b, name in zip(out, exp, ["mse_rtn", "mse_gptq", "rtn_q", "gptq_q"]):
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), f"{name} mismatch"


def check_group_size(solve):
    """不同 group_size 应改变分位数张量长度不变、数值合理。"""
    torch.manual_seed(61)
    W = torch.randn(4, 64)
    H = torch.eye(64) * 0.1 + 0.01 * torch.randn(64, 64).triu()
    H = H @ H.T
    X = torch.randn(2, 64)
    _, _, rtn_q, gptq_q = solve(W, H, X, 4, 0.01, 16)
    assert rtn_q.shape == (4,) and gptq_q.shape == (4,)
    assert torch.all(rtn_q[1:] >= rtn_q[:-1] - 1e-6), "分位数应非降"
    assert torch.all(gptq_q[1:] >= gptq_q[:-1] - 1e-6), "分位数应非降"


def check_bits_range(solve):
    """bits=2 时量化网格更粗，误差应不小于 bits=4。"""
    torch.manual_seed(67)
    W = torch.randn(4, 32)
    H = torch.eye(32) * 0.1
    X = torch.randn(2, 32)
    mse_rtn_2, _, _, _ = solve(W, H, X, 2, 0.01, 8)
    mse_rtn_4, _, _, _ = solve(W, H, X, 4, 0.01, 8)
    assert mse_rtn_2 >= mse_rtn_4 * 0.99, "2-bit 误差不应显著小于 4-bit"
