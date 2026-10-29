"""参考答案：把前向索引式原样反转，dx 用 scatter-add、dw 用同一套索引做点积。

前向是「带 stride/padding 的相关」，反向的两条路共享同一个 (u,v,k,l) → (ii,jj) 映射：
dx[b,ci,ii,jj] += dY[b,co,u,v] · w[co,ci,k,l]（等价于与翻转核做全卷积），
dw[co,ci,k,l] += dY[b,co,u,v] · x[b,ci,ii,jj]（对 batch 与窗口求和）。
越界的 (ii,jj) 两边都必须丢弃，padding 只影响前向读数、不产生 dx 的额外行。
"""

import torch


def solve(dY, x, weight, stride=1, padding=0):
    if x.dim() != 4 or weight.dim() != 4 or dY.dim() != 4:
        raise ValueError("x/weight/dY 必须都是四维张量")
    b, cin, hin, win = x.shape
    cout, cin_w, kh, kw = weight.shape
    if cin_w != cin:
        raise ValueError("weight 的 Cin 必须与 x 的通道数一致")
    if dY.shape[0] != b or dY.shape[1] != cout:
        raise ValueError("dY 的 batch/Cout 必须与 x/weight 一致")
    s, p = int(stride), int(padding)
    if s <= 0:
        raise ValueError("stride 必须为正")
    if p < 0:
        raise ValueError("padding 不得为负")
    hout, wout = dY.shape[2], dY.shape[3]
    if hout != (hin + 2 * p - kh) // s + 1 or wout != (win + 2 * p - kw) // s + 1:
        raise ValueError("dY 的空间尺寸与 x/weight/stride/padding 不符")

    xd = x.double()
    wd = weight.double()
    yd = dY.double()
    dx = torch.zeros_like(xd)
    dw = torch.zeros_like(wd)

    for co in range(cout):
        for ci in range(cin):
            for u in range(hout):
                i0 = u * s - p
                for v in range(wout):
                    j0 = v * s - p
                    gy = yd[:, co, u, v]
                    for k in range(kh):
                        ii = i0 + k
                        if ii < 0 or ii >= hin:
                            continue
                        for l in range(kw):
                            jj = j0 + l
                            if jj < 0 or jj >= win:
                                continue
                            dx[:, ci, ii, jj] += gy * wd[co, ci, k, l]
                            dw[co, ci, k, l] += (gy * xd[:, ci, ii, jj]).sum()

    return dx.to(x.dtype), dw.to(weight.dtype)
