import math

import torch


def _design(up, down, taps, beta):
    """原型低通，数组下标 j 对应「有符号偏移」j - (taps-1)/2。"""
    # 截止频率取「上采样后栅格」上的 Nyquist/max(L,M)：既挡住镜像又挡住混叠
    fc = 0.5 / max(up, down)
    center = (taps - 1) // 2
    j = torch.arange(taps, dtype=torch.float64) - center
    h = 2.0 * fc * torch.sinc(2.0 * fc * j)
    h = h * torch.kaiser_window(taps, periodic=False, beta=float(beta), dtype=torch.float64)
    return h * (up / h.sum())


def solve(x, up, down, taps=63, beta=8.0):
    if not isinstance(x, torch.Tensor) or x.dim() != 1:
        raise ValueError("x 必须是一维张量")
    if int(up) < 1 or int(down) < 1:
        raise ValueError("up / down 必须为正整数")
    if int(taps) < 3 or int(taps) % 2 == 0:
        raise ValueError("taps 必须是 >= 3 的奇数")
    if float(beta) < 0:
        raise ValueError("beta 不能为负")

    up, down, taps = int(up), int(down), int(taps)
    center = (taps - 1) // 2
    sig = x.to(torch.float64)
    total = int(sig.numel())
    if total < 1:
        raise ValueError("x 不能为空")

    h = _design(up, down, taps, beta)
    out_len = ((total - 1) * up) // down + 1

    y = torch.zeros(out_len, dtype=torch.float64)
    for m in range(out_len):
        base = m * down
        # 需要 0 <= base - n*up + center <= taps-1
        n_lo = max(0, -((-(base + center - (taps - 1))) // up))
        n_hi = min(total - 1, (base + center) // up)
        acc = 0.0
        for n in range(n_lo, n_hi + 1):
            acc += float(h[base - n * up + center]) * float(sig[n])
        y[m] = acc

    return y.to(torch.float32)
