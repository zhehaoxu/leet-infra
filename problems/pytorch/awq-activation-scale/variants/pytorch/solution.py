import torch


def _rtn_per_tensor(x, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = float(x.abs().max()) or 1.0
    q = torch.clamp(torch.round(x / scale * qmax), -qmax - 1, qmax)
    return q / qmax * scale


def _awq_wdq(W, x_bar, alpha, bits):
    raw = x_bar.pow(alpha)
    s = raw / raw.mean()
    return _rtn_per_tensor(W * s.unsqueeze(0), bits) / s.unsqueeze(0)


def solve(W, x_bar, alphas, bits):
    """AWQ：逐通道缩放 + per-tensor RTN，按 x_bar^2 加权输出误差网格搜索最优 alpha。"""
    best, best_err = None, float("inf")
    for a in alphas:
        Wdq = _awq_wdq(W, x_bar, a, bits)
        err = float(((W - Wdq).pow(2) * (x_bar.unsqueeze(0) ** 2)).sum())
        if err < best_err - 1e-12:      # 严格小于 → 并列保留先出现者
            best_err, best = err, a
    return float(best), _awq_wdq(W, x_bar, best, bits)
