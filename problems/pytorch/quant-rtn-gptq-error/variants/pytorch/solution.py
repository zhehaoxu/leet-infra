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


def _per_group_mae(W, W_hat, group_size):
    O, C = W.shape
    err = (W - W_hat).abs().view(O, C // group_size, group_size)
    return err.mean(dim=2)


def solve(W, H, X, bits, damp, group_size):
    """RTN 与 GPTQ 输出 MSE 及逐组误差分位数对比。"""
    qmax = 2 ** (bits - 1) - 1
    scale = float(W.abs().max()) or 1.0
    W_rtn = _quantize(W, scale, qmax)
    W_gptq = _gptq(W, H, bits, damp)

    Y_true = X @ W.T
    Y_rtn = X @ W_rtn.T
    Y_gptq = X @ W_gptq.T

    mse_rtn = float(((Y_true - Y_rtn) ** 2).mean())
    mse_gptq = float(((Y_true - Y_gptq) ** 2).mean())

    rtn_mae = _per_group_mae(W, W_rtn, group_size).flatten()
    gptq_mae = _per_group_mae(W, W_gptq, group_size).flatten()
    probs = torch.tensor([0.25, 0.5, 0.75, 0.9], dtype=W.dtype, device=W.device)
    rtn_q = torch.quantile(rtn_mae, probs).to(W.dtype)
    gptq_q = torch.quantile(gptq_mae, probs).to(W.dtype)

    return (torch.tensor(mse_rtn, dtype=W.dtype),
            torch.tensor(mse_gptq, dtype=W.dtype),
            rtn_q, gptq_q)
