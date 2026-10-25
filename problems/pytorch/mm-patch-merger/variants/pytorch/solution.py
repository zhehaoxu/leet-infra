"""参考答案：2×2 空间 shuffle + concat（不含带权重 Linear）。"""

import torch


def solve(image_embeds, grid_h, grid_w):
    d = image_embeds.shape[1]
    return (image_embeds.view(grid_h // 2, 2, grid_w // 2, 2, d)
                            .permute(0, 2, 1, 3, 4)
                            .reshape(-1, 4 * d)
                            .contiguous())
