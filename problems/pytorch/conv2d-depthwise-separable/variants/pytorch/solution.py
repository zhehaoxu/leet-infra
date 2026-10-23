"""参考答案：深度可分离卷积（depthwise + pointwise，MobileNet 式）。

先做 depthwise：每个输入通道用各自 (kh,kw) 核单独卷积（groups=C），padding=kh//2 保尺寸；
再做 pointwise：1x1 卷积跨通道线性组合到 out_c。kh 取奇数。
"""

import torch


def solve(x, dw_weight, pw_weight, dw_bias=None, pw_bias=None):
    n, c, h, w = x.shape
    kh, kw = dw_weight.shape[2], dw_weight.shape[3]
    p = kh // 2
    xp = torch.nn.functional.pad(x, (p, p, p, p))
    oh, ow = h, w
    dw = x.new_zeros((n, c, oh, ow))
    for ci in range(c):                                          # depthwise：逐通道独立核
        for i in range(oh):
            for j in range(ow):
                region = xp[:, ci, i:i + kh, j:j + kw]           # (n, kh, kw)
                dw[:, ci, i, j] = (region * dw_weight[ci, 0]).sum(dim=(1, 2))
    if dw_bias is not None:
        dw = dw + dw_bias.view(1, -1, 1, 1)
    cout = pw_weight.shape[0]
    pw = pw_weight.reshape(cout, c)                              # (out_c, C) 1x1
    dm = dw.permute(0, 2, 3, 1).reshape(-1, c)                   # (n*oh*ow, C)
    y = (dm @ pw.t()).reshape(n, oh, ow, cout).permute(0, 3, 1, 2)
    if pw_bias is not None:
        y = y + pw_bias.view(1, -1, 1, 1)
    return y.contiguous()
