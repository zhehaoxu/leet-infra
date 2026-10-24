"""参考答案：RoI Pooling（纯 torch，量化分 bin + 窗口 max）。

把每个 ROI 按 spatial_scale 映射并 floor 量化到特征图整数格点，用整数地板除把矩形切成
P×P 个 bin，空 bin 记 0，否则裁剪到边界后逐通道取窗口最大值。与题面描述、独立 numpy 实现一致。
"""

import math

import torch


def solve(features, rois, pooled_size, spatial_scale):
    N, C, H, W = features.shape
    K = rois.shape[0]
    out = features.new_zeros((K, C, pooled_size, pooled_size))
    for k in range(K):
        b = int(rois[k, 0].item())
        x1, y1, x2, y2 = [rois[k, i].item() for i in range(1, 5)]
        fx1 = math.floor(x1 * spatial_scale); fy1 = math.floor(y1 * spatial_scale)
        fx2 = math.floor(x2 * spatial_scale); fy2 = math.floor(y2 * spatial_scale)
        for i in range(pooled_size):
            ylo = fy1 + (fy2 - fy1) * i // pooled_size
            yhi = fy1 + (fy2 - fy1) * (i + 1) // pooled_size
            for j in range(pooled_size):
                xlo = fx1 + (fx2 - fx1) * j // pooled_size
                xhi = fx1 + (fx2 - fx1) * (j + 1) // pooled_size
                if yhi <= ylo or xhi <= xlo:
                    continue
                yl, yh = max(0, ylo), min(H, yhi)
                xl, xh = max(0, xlo), min(W, xhi)
                if yh <= yl or xh <= xl:
                    continue
                region = features[b, :, yl:yh, xl:xh]
                out[k, :, i, j] = region.reshape(C, -1).amax(dim=1)
    return out
