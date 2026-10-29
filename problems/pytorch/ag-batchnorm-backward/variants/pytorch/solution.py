"""参考答案：把三段式合并成沿 batch 轴的一次双投影，避免 1/N 被算两遍。

朴素推导会把 dX 写成三项相加：a/std、−(Σa)/(N·std³)·(x−mu)、−(Σa·(x−mu))/(N·std³)·... 
合并后正是 LN 的同款结构，只是求和轴换成了 batch：
dX = (a − mean_N(a) − xhat·mean_N(a·xhat)) / std。
差别在于「归一化统计量来自 batch」⇒ 同一列的所有样本互相耦合，而 LN 是同一行内部耦合。
"""

import torch


def solve(x, gamma, beta, dY, eps=1e-5):
    if eps is None or float(eps) < 0:
        raise ValueError("eps 不得为负")
    if x.dim() != 2:
        raise ValueError("x 必须是 (N, C) 二维张量")
    if gamma.shape != beta.shape or gamma.shape != (x.shape[1],):
        raise ValueError("gamma/beta 必须是长度为 C 的一维张量")
    if tuple(x.shape) != tuple(dY.shape):
        raise ValueError("dY 与 x 形状必须一致")

    xd = x.double()
    gd = gamma.double()
    dYd = dY.double()

    mu = xd.mean(dim=0, keepdim=True)
    xc = xd - mu
    var = (xc * xc).mean(dim=0, keepdim=True)
    std = (var + float(eps)).sqrt()
    xhat = xc / std

    a = dYd * gd
    dX = (a - a.mean(dim=0, keepdim=True)
          - xhat * (a * xhat).mean(dim=0, keepdim=True)) / std

    dGamma = (dYd * xhat).sum(dim=0)
    dBeta = dYd.sum(dim=0)
    return dX.to(x.dtype), dGamma.to(gamma.dtype), dBeta.to(beta.dtype)
