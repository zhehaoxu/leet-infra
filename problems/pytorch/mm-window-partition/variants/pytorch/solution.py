"""参考答案：Swin window partition（view+permute 重排）。"""

import torch


def solve(x, window):
    B, H, W, C = x.shape
    nr, nc = H // window, W // window
    return (x.view(B, nr, window, nc, window, C)
              .permute(0, 1, 3, 2, 4, 5)
              .reshape(B * nr * nc, window, window, C)
              .contiguous())
