"""参考答案：ViT patchify（reshape+permute 纯重排）。"""

import torch


def solve(image, patch):
    B, C, H, W = image.shape
    nr, nc = H // patch, W // patch
    return (image.reshape(B, C, nr, patch, nc, patch)
                 .permute(0, 2, 4, 1, 3, 5)
                 .reshape(B, nr * nc, C * patch * patch)
                 .contiguous())
