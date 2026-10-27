"""参考答案：通道↔空间双射重排 —— 纯 reshape/permute 实现（不调用 F.pixel_*）。"""

import torch


def _d2s(x, r):
    """(N, C·r², H, W) → (N, C, H·r, W·r)：out[n,c,h*r+i,w*r+j] = x[n,c*r²+i*r+j,h,w]。"""
    n, cr2, h, w = x.shape
    c = cr2 // (r * r)
    # [n, c, i, j, h, w] --permute--> [n, c, h, i, w, j] --reshape--> [n, c, h*r+i, w*r+j]
    return x.reshape(n, c, r, r, h, w).permute(0, 1, 4, 2, 5, 3).reshape(n, c, h * r, w * r)


def _s2d(x, r):
    """(N, C, H, W) → (N, C·r², H/r, W/r)：out[n,c*r²+i*r+j,h,w] = x[n,c,h*r+i,w*r+j]。"""
    n, c, h, w = x.shape
    hb, wb = h // r, w // r
    # [n, c, hb, i, wb, j] --permute--> [n, c, i, j, hb, wb] --reshape--> 通道序 c*r²+i*r+j
    return x.reshape(n, c, hb, r, wb, r).permute(0, 1, 3, 5, 2, 4).reshape(n, c * r * r, hb, wb)


def solve(x, r, mode="shuffle"):
    if not isinstance(r, int) or isinstance(r, bool) or r < 1:
        raise ValueError(f"r 必须是 int 类型且 >= 1，got {r!r}")
    if x.dim() != 4:
        raise ValueError(f"x 必须是 4 维 (N, C, H, W)，got {tuple(x.shape)}")
    n, c, h, w = x.shape
    if mode in ("shuffle", "d2s"):
        if c % (r * r) != 0:
            raise ValueError(f"通道数 {c} 不能被 r²={r * r} 整除")
        return _d2s(x, r)
    if mode in ("unshuffle", "s2d"):
        if h % r != 0 or w % r != 0:
            raise ValueError(f"H={h}、W={w} 必须都能被 r={r} 整除")
        return _s2d(x, r)
    raise ValueError(f"unknown mode: {mode!r}")
