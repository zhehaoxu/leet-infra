"""参考答案：末轴上的融合式 dX，外加把前导轴整体折叠的 dGamma/dBeta。

把 xhat = (x − mu)/std 的两条依赖（mu 与 std 都由 x 决定）合起来写，就得到
dX = (g − mean(g) − xhat·mean(g·xhat))/std：第一项是直通，第二项是「减均值」带来的
投影回零，第三项是「除标准差」带来的沿 xhat 方向的投影。分开推很容易重复计入。
dGamma/dBeta 是对所有前导轴（batch/序列/头）求和，用 reshape 折叠成二维后一次 sum。
"""

import torch


def solve(x, gamma, beta, dY, eps=1e-5):
    if x.shape[-1] != gamma.shape[0] or gamma.shape != beta.shape:
        raise ValueError("gamma/beta 必须是与末轴等长的一维张量")
    if tuple(x.shape) != tuple(dY.shape):
        raise ValueError("dY 与 x 形状必须一致")
    if x.dim() < 1:
        raise ValueError("x 至少是一维")

    xd = x.double()
    gd_ = gamma.double()
    dYd = dY.double()
    h = xd.shape[-1]

    mu = xd.mean(dim=-1, keepdim=True)
    xc = xd - mu
    var = (xc * xc).mean(dim=-1, keepdim=True)
    std = (var + float(eps)).sqrt()
    xhat = xc / std

    g = dYd * gd_
    dX = (g - g.mean(dim=-1, keepdim=True)
          - xhat * (g * xhat).mean(dim=-1, keepdim=True)) / std

    flat_y = dYd.reshape(-1, h)
    flat_hat = xhat.reshape(-1, h)
    dGamma = (flat_y * flat_hat).sum(dim=0)
    dBeta = flat_y.sum(dim=0)

    return dX.to(x.dtype), dGamma.to(gamma.dtype), dBeta.to(beta.dtype)
