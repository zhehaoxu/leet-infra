"""参考答案：切片折叠算模长，dv 取「减掉法向分量后的投影」。

对第 i 片：w = g·v/n，n = ‖v‖。dg = ⟨dW, v/n⟩；对 v 求导时，dW 里沿 v 方向的分量
已经被 dg 吸收了（沿 v 缩放 v 不改变 w），所以剩下的部分必须整体除以 n 再乘 g：
dV = (g/n)·(dW − v̂·dg)。少了减法就会得到与 autograd 不一致的「双重计入」结果。
"""

import torch


def solve(V, g, dW):
    if V.dim() < 1 or dW.shape != V.shape:
        raise ValueError("dW 必须与 V 同形")
    out = V.shape[0]
    if g.shape != (out,):
        raise ValueError("g 必须是长度为 V.shape[0] 的一维张量")

    vd = V.double()
    gd = g.double()
    dWd = dW.double()
    flat = vd.reshape(out, -1)
    dflat = dWd.reshape(out, -1)

    n = flat.norm(dim=1)
    if bool((n <= 0).any()):
        raise ValueError("存在零向量的 weight_norm 切片，方向无定义")
    vhat = flat / n.unsqueeze(1)

    dg = (dflat * vhat).sum(dim=1)
    dV = ((gd / n).unsqueeze(1) * (dflat - vhat * dg.unsqueeze(1))).reshape(V.shape)

    return dg.to(g.dtype), dV.to(V.dtype)
