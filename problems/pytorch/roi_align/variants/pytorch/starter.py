import torch


def solve(features, rois, pooled_size, spatial_scale, sampling_ratio, aligned):
    # TODO: RoI Align（纯 torch，不量化 + 网格双线性平均）。
    # features:(N,C,H,W), rois:(K,5) 每行 [b, X1,Y1,X2,Y2]（原图坐标 float）。
    # 对每个 ROI：x1=X1*s,...（不 floor）；若 aligned 则四项各 -0.5；bin_w=(x2-x1)/P, bin_h=(y2-y1)/P。
    #   第 (i,j) bin 布 r×r 网格，采样点 fx=x_start+(b+0.5)*bin_w/r（fy 同理）。
    #   每采样点逐通道双线性插值：四邻域 (floor+1) 加权，越界格点记 0；bin = r*r 个采样点平均。
    # 输出 (K,C,P,P)，建议 float64 累加。正确性以独立 numpy 参考实现为准（详见题面规则）。
    raise NotImplementedError
