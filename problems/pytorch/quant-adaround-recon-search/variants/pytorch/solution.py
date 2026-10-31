"""参考解：AdaRound 的确定性逐元素候选搜索（CPU，内部 float64）。"""

import torch

K0 = -0.1
K1 = 1.1
EPS = 1e-6


def _decode(h, res_log, res):
    """软解码：_h = floor(h*res)/res（夹到 [eps, 1-eps]），alpha = sigmoid((logit(_h)+log res)/2)。"""
    ladder = torch.floor(h * res) / res
    ladder = torch.clamp(ladder, EPS, 1.0 - EPS)
    logit = torch.log(ladder) - torch.log(1.0 - ladder)
    return torch.sigmoid((logit + res_log) / 2.0)


def _objective(alpha, z, X64, scale64, n_cols, beta, p):
    """L = Σ_o ||X (ŵ_o - w_o)||² / C + β · mean(|α - 0.5|^p)，其中 ŵ - w = (α - z)·scale。"""
    delta = (alpha - z) * scale64
    err = (X64 @ delta.T).pow(2).sum() / n_cols
    reg = beta * (alpha - 0.5).abs().pow(p).mean()
    return err + reg


def solve(W, X, scale, num_iters, steps, lam):
    if int(num_iters) < 1:
        raise ValueError("num_iters must be >= 1")
    if int(steps) < 1:
        raise ValueError("steps must be >= 1")
    if float(scale) <= 0.0:
        raise ValueError("scale must be > 0")
    if not (float(lam) > 0.0 and float(lam) != 1.0):
        raise ValueError("lam must be > 0 and != 1")
    if W.dim() != 2 or X.dim() != 2 or W.shape[1] != X.shape[1]:
        raise ValueError("W must be 2-D, X must be 2-D and share W's input channels")
    if W.dtype != X.dtype:
        raise ValueError("W and X must have the same dtype")
    if min(int(W.shape[0]), int(W.shape[1]), int(X.shape[0])) < 1:
        raise ValueError("empty W / X is not supported")

    dtype = W.dtype if W.dtype in (torch.float32, torch.float64) else torch.float32
    W64 = W.detach().to(torch.float64).contiguous()
    X64 = X.detach().to(torch.float64).contiguous()
    n_rows, n_cols = W64.shape
    scale64 = torch.tensor(float(scale), dtype=torch.float64)

    res = float(2 ** int(num_iters))
    res_log = float(torch.tensor(res, dtype=torch.float64).log())
    total = int(steps)
    log_lam = float(torch.tensor(float(lam), dtype=torch.float64).log())
    betas = [(1.0 + (i + 1) / total) / log_lam for i in range(total)]
    powers = [1.0 + (i + 1) / total for i in range(total)]
    offsets = [0.0, 1.0 / res, -1.0 / res, 1.0, -1.0]

    w_norm = W64 / scale64
    q = torch.floor(w_norm)
    z = w_norm - q                       # 归一化残差，[0, 1)
    r = W64 - q * scale64                # 绝对残差，[0, scale)
    h = torch.clamp((z - K0) / (K1 - K0), min=0.0, max=1.0)
    h = torch.where(q == 0, torch.ones_like(h), h)

    frozen = torch.zeros_like(W64, dtype=torch.bool)
    chosen = torch.zeros_like(W64)
    soft_loss = []
    for i in range(total):
        for k in range(n_rows * n_cols):
            o, j = divmod(k, n_cols)
            if bool(frozen[o, j]):
                continue
            best_value = None
            best_index = 0
            for c, off in enumerate(offsets):
                trial = h.clone()
                trial[o, j] = min(1.0, max(0.0, float(h[o, j]) + off))
                value = float(_objective(_decode(trial, res_log, res), z, X64,
                                         scale64, n_cols, betas[i], powers[i]))
                if best_value is None or value < best_value:
                    best_value = value
                    best_index = c
            off = offsets[best_index]
            h[o, j] = min(1.0, max(0.0, float(h[o, j]) + off))
            chosen[o, j] = off
            frozen[o, j] = abs(float(h[o, j]) - 0.5) > 0.5 / res
        soft_loss.append(float(_objective(_decode(h, res_log, res), z, X64, scale64,
                                          n_cols, betas[i], powers[i])))

    alpha = _decode(h, res_log, res)
    mask = (h - 0.5).abs() <= 0.5 / res + EPS
    ladder = torch.floor(h * res) / res
    alpha_hard = torch.where(mask, (ladder >= 0.5).to(torch.float64), torch.round(alpha))

    searched = (q + alpha_hard) * scale64
    mse = float((X64 @ (searched - W64).T).pow(2).sum() / n_cols)

    return {
        "q": q.to(torch.int64),
        "r": r.to(dtype),
        "h": h.to(dtype),
        "alpha": alpha.to(dtype),
        "betas": torch.tensor(betas, dtype=dtype),
        "soft_loss": torch.tensor(soft_loss, dtype=dtype),
        "searched_weight": searched.to(dtype),
        "chosen_offset": chosen.to(dtype),
        "searched_mse": torch.tensor(mse, dtype=dtype),
    }
