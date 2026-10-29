import torch


def _quant_sym(x, qmax):
    """逐行对称整数量化（dim=1），零行反量化为 0。"""
    amax = x.abs().amax(dim=1, keepdim=True)
    scale = amax / qmax
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.clamp(torch.round(x / safe), -qmax, qmax)
    return q * safe


def solve(X, W, alphas, bits):
    if bits < 2 or bits > 8:
        raise ValueError("bits must be in [2, 8]")
    Xf = X.to(torch.float64)
    Wf = W.to(torch.float64)
    qmax = 2 ** (bits - 1) - 1
    sx = Xf.abs().amax(dim=0)
    sw = Wf.abs().amax(dim=0)
    usable = (sx > 0) & (sw > 0)
    Y = Xf @ Wf.T

    best = None
    mses = []
    for alpha in alphas:
        a = float(alpha)
        s = torch.ones_like(sx)
        s[usable] = sx[usable] ** a / sw[usable] ** (1.0 - a)
        Xd = _quant_sym(Xf / s, qmax)
        Wd = _quant_sym(Wf * s, qmax)
        mse = float(((Y - Xd @ Wd.T) ** 2).mean())
        mses.append(mse)
        if best is None or mse < best[0]:
            best = (mse, a, s)

    return best[1], best[2].to(torch.float32), torch.tensor(mses, dtype=torch.float32)
