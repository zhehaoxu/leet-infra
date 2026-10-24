"""参考答案：双线性网格采样 F.grid_sample（mode='bilinear', padding_mode='zeros',
align_corners=False）。

约定：grid 最后一维为 (x, y)，取值 [-1, 1]；(-1,-1) 对应输入左上角像素中心、
(1,1) 对应右下角像素中心。align_corners=False 时归一化坐标到像素坐标的映射为
    x_pix = 0.5 * ((g + 1) * W - 1)
对每个输出位置，取周围 4 个整数像素做双线性加权；越界处按 0 填充（padding）。
"""

import torch


def solve(x, grid):
    n, c, h, w = x.shape
    _, ho, wo, _ = grid.shape
    # 四周补一圈 0，越界的邻居索引自动落到补零区（padding_mode='zeros'）
    xp = torch.zeros((n, c, h + 2, w + 2), dtype=x.dtype, device=x.device)
    xp[:, :, 1:h + 1, 1:w + 1] = x

    gx = grid[..., 0]  # (n, ho, wo) 宽度方向归一化坐标
    gy = grid[..., 1]  # (n, ho, wo) 高度方向归一化坐标
    # align_corners=False 的坐标变换
    ix = 0.5 * (gx + 1) * w - 0.5
    iy = 0.5 * (gy + 1) * h - 0.5

    x0 = torch.floor(ix).long()
    y0 = torch.floor(iy).long()
    # +1 变成补零张量里的索引
    xa, ya = x0 + 1, y0 + 1
    xb, yb = xa + 1, ya + 1

    wx = ix - x0.to(x.dtype)  #  fractional 权重
    wy = iy - y0.to(x.dtype)

    out = x.new_zeros((n, c, ho, wo))
    for b in range(n):
        for oy in range(ho):
            for ox in range(wo):
                ya_, yb_ = int(ya[b, oy, ox]), int(yb[b, oy, ox])
                xa_, xb_ = int(xa[b, oy, ox]), int(xb[b, oy, ox])
                fy, fx = wy[b, oy, ox], wx[b, oy, ox]
                top = xp[b, :, ya_, xa_] * (1 - fx) + xp[b, :, ya_, xb_] * fx
                bot = xp[b, :, yb_, xa_] * (1 - fx) + xp[b, :, yb_, xb_] * fx
                out[b, :, oy, ox] = top * (1 - fy) + bot * fy
    return out
