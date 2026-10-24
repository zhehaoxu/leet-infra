import torch


def solve(features, rois, pooled_size, spatial_scale):
    # TODO: RoI Pooling。features:(N,C,H,W), rois:(K,5) 每行 [b, x1,y1,x2,y2]（原图坐标 float）。
    # 对每个 ROI：
    #   fx1=floor(x1*s), fy1=floor(y1*s), fx2=floor(x2*s), fy2=floor(y2*s)
    #   bin(i,j) 行范围 ylo=fy1+(fy2-fy1)*i//P, yhi=fy1+(fy2-fy1)*(i+1)//P；x 同理（整数地板除）
    #   空 bin(yhi<=ylo 或 xhi<=xlo)→0；否则裁剪到 [0,H]x[0,W] 后每通道取窗口 max
    # 输出 (K, C, P, P)。正确性以独立 numpy 参考实现为准（详见题面规则）。
    raise NotImplementedError
