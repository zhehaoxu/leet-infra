import torch


def solve(x, dw_weight, pw_weight, dw_bias=None, pw_bias=None):
    # TODO: 深度可分离卷积 = depthwise + pointwise。
    # 1) depthwise: dw_weight (C,1,kh,kh), padding=kh//2, groups=C（逐通道独立核）；加 dw_bias。
    # 2) pointwise: pw_weight (Cout,C,1,1)，1x1 跨通道线性组合；加 pw_bias。
    raise NotImplementedError
