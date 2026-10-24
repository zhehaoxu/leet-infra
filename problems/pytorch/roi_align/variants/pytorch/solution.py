"""参考答案：RoI Align（纯 torch，不量化 + 网格双线性插值平均）。

全程保留浮点：ROI 端点按 spatial_scale 映射（aligned 再各减 0.5），把浮点矩形均分成 P×P 个 bin，
每个 bin 内布 sampling_ratio×sampling_ratio 网格，逐点做双线性插值（越界格点记 0）后取平均。
与题面描述、独立 numpy 实现一致。
"""

import math

import torch


def _bilinear(feat_c, y, x):
    """在 (C,H,W) 的单个通道图 feat_c 上对浮点坐标 (y,x) 双线性插值；越界格点贡献 0。"""
    H, W = feat_c.shape
    y0 = math.floor(y); x0 = math.floor(x)
    y1 = y0 + 1; x1 = x0 + 1
    ly = y - y0; lx = x - x0

    def val(a, b):
        if 0 <= a < H and 0 <= b < W:
            return feat_c[a, b].item()
        return 0.0

    return ((1 - ly) * (1 - lx) * val(y0, x0) + (1 - ly) * lx * val(y0, x1) +
            ly * (1 - lx) * val(y1, x0) + ly * lx * val(y1, x1))


def solve(features, rois, pooled_size, spatial_scale, sampling_ratio, aligned):
    N, C, H, W = features.shape
    K = rois.shape[0]
    out = torch.zeros((K, C, pooled_size, pooled_size), dtype=torch.float64)
    r = max(1, int(sampling_ratio))
    f64 = features.double()
    for k in range(K):
        b = int(rois[k, 0].item())
        x1 = rois[k, 1].item() * spatial_scale
        y1 = rois[k, 2].item() * spatial_scale
        x2 = rois[k, 3].item() * spatial_scale
        y2 = rois[k, 4].item() * spatial_scale
        if aligned:
            x1 -= 0.5; y1 -= 0.5; x2 -= 0.5; y2 -= 0.5
        bin_w = (x2 - x1) / pooled_size
        bin_h = (y2 - y1) / pooled_size
        fmap = f64[b]  # (C,H,W)
        for i in range(pooled_size):
            ys = y1 + i * bin_h
            for j in range(pooled_size):
                xs = x1 + j * bin_w
                acc = torch.zeros(C, dtype=torch.float64)
                for sy in range(r):
                    fy = ys + (sy + 0.5) * bin_h / r
                    for sx in range(r):
                        fx = xs + (sx + 0.5) * bin_w / r
                        for c in range(C):
                            acc[c] += _bilinear(fmap[c], fy, fx)
                out[k, :, i, j] = acc / (r * r)
    return out
