import torch


def solve(scores, rois, k):
    # TODO: R-FCN 的位置敏感 RoI 池化（PS-RoI pooling）。
    #   - scores 形状 (C, k*k, H, W)：每个空间位置有 k*k 个通道，通道 i*k+j 只负责 bin (i, j)。
    #   - rois 是 (N, 4) 的整数特征坐标 [x1, y1, x2, y2]（左闭右开），先 clip 到 [0, W] / [0, H]。
    #   - bin 划分用整数整除：行区间 [y1 + i*h//k, y1 + (i+1)*h//k)，列区间同理；
    #     h = max(y2 - y1, 0)，w = max(x2 - x1, 0)。
    #   - 每个 bin 的值 = 对应通道在该子区域内的算术平均；子区域为空（cnt == 0）时值为 0.0。
    #   - 最终得分 = k*k 个 bin 值的**等权**平均（空 bin 的 0 也参与平均）。
    #   返回 {"pooled": (N, C), "bins": (N, C, k*k), "bin_sizes": (N, k*k),
    #         "bounds": (N, 4), "empty": (N, k*k)}。
    raise NotImplementedError
