"""SPP-Net 金字塔池化账（solution）。

转写 BVLC/caffe src/caffe/layers/spp_layer.cpp GetPoolingParam +
pooling_layer.cpp MaxPoolForward：kernel=ceil(H/k)、stride=kernel、
pad=(kernel·k−H+1)//2，bin 边界 clip 后窗口 max，按 level 拼接。
"""

import math


def solve(feature, bin_counts):
    if not isinstance(feature, list) or not feature or not all(
        isinstance(row, list) and row and all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in row
        )
        for row in feature
    ):
        raise ValueError("feature must be a non-empty 2-D list of finite floats")
    h = len(feature)
    w = len(feature[0])
    if any(len(row) != w for row in feature):
        raise ValueError("feature rows must be uniform width")
    if not isinstance(bin_counts, list) or not bin_counts or not all(
        isinstance(k, int) and k >= 1 for k in bin_counts
    ):
        raise ValueError("bin_counts must be a non-empty list of ints >= 1")
    kmax = max(bin_counts)
    if h < kmax or w < kmax:
        raise ValueError("H, W must be >= max bin count (no empty bins)")

    pooled = []
    shapes = []
    kernels = []
    pads = []
    for k in bin_counts:
        # GetPoolingParam：每级 kernel=ceil(H/k)（H、W 各自算），stride=kernel，
        # pad=(remainder+1)//2 对称分到上下
        kernel_h = math.ceil(h / k)
        kernel_w = math.ceil(w / k)
        pad_h = (kernel_h * k - h + 1) // 2
        pad_w = (kernel_w * k - w + 1) // 2
        kernels.append([kernel_h, kernel_w])
        pads.append([pad_h, pad_w])
        shapes.append([k, k])
        for ph in range(k):
            for pw in range(k):
                # MaxPoolForward 边界：可负 clip 0、超界 clip H/W
                hs = max(0, ph * kernel_h - pad_h)
                ws = max(0, pw * kernel_w - pad_w)
                he = min(hs + kernel_h, h)
                we = min(ws + kernel_w, w)
                best = -math.inf
                for r in range(hs, he):
                    for c in range(ws, we):
                        if feature[r][c] > best:
                            best = feature[r][c]
                pooled.append(best)

    return {"pooled": pooled, "shapes": shapes,
            "kernels": kernels, "pads": pads}
