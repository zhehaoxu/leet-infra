"""参考答案：dX = dY @ W，dW = 折叠所有前导轴后的 dY.T @ X。

前向写成 Y = X @ W.T 是为了贴住 nn.Linear 的参数约定（weight 是 (out, in)），
于是 dX 与 dW 各自只需一次矩阵乘：dX 右乘 W（不需要转置），dW 需要把 dY 转置后
左乘 X。关键手法是把 X、dY 都 reshape 成二维 (M_total, ·) 再乘一次，等价于对
所有 batch 轴求和——显式 for 循环累加会慢且容易漏轴。
"""

import torch


def solve(dY, X, W):
    if W.dim() != 2:
        raise ValueError("W 必须是 (N, K) 二维权重")
    if dY.dim() < 2 or X.dim() < 2:
        raise ValueError("dY 与 X 至少是二维（..., M, ·）")
    if X.shape[-2] != dY.shape[-2]:
        raise ValueError("dY 与 X 的 M 轴长度不一致")
    if X.shape[-1] != W.shape[1]:
        raise ValueError("X 的末轴必须等于 W 的 in_features")
    if dY.shape[-1] != W.shape[0]:
        raise ValueError("dY 的末轴必须等于 W 的 out_features")
    if X.shape[:-2] != dY.shape[:-2]:
        raise ValueError("dY 与 X 的前导 batch 轴必须一致")

    xd = X.double()
    yd = dY.double()
    wd = W.double()

    dX = yd @ wd
    xf = xd.reshape(-1, xd.shape[-1])
    yf = yd.reshape(-1, yd.shape[-1])
    dW = yf.t() @ xf

    return dX.to(X.dtype), dW.to(W.dtype)
