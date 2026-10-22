"""参考答案：双线性插值上/下采样 —— 坐标映射后取四邻域加权混合。"""

import torch


def solve(x: torch.Tensor, size, align_corners: bool = False) -> torch.Tensor:
    if isinstance(size, int):
        size = (size, size)
    oh, ow = size
    n, c, h, w = x.shape
    dtype = x.dtype
    if align_corners:
        sy = (h - 1) / max(oh - 1, 1)
        sx = (w - 1) / max(ow - 1, 1)
        iy = torch.arange(oh, dtype=dtype) * sy
        ix = torch.arange(ow, dtype=dtype) * sx
    else:
        sy = h / oh
        sx = w / ow
        iy = (torch.arange(oh, dtype=dtype) + 0.5) * sy - 0.5
        ix = (torch.arange(ow, dtype=dtype) + 0.5) * sx - 0.5
    iy = iy.clamp(0, h - 1)
    ix = ix.clamp(0, w - 1)
    y0 = iy.floor().long()
    x0 = ix.floor().long()
    y1 = (y0 + 1).clamp(max=h - 1)
    x1 = (x0 + 1).clamp(max=w - 1)
    wy = (iy - y0.to(dtype)).view(1, 1, -1, 1)
    wx = (ix - x0.to(dtype)).view(1, 1, 1, -1)
    # 四角取值：先按行索引再按列索引
    v00 = x[:, :, y0][:, :, :, x0]
    v01 = x[:, :, y0][:, :, :, x1]
    v10 = x[:, :, y1][:, :, :, x0]
    v11 = x[:, :, y1][:, :, :, x1]
    return (1 - wy) * (1 - wx) * v00 + (1 - wy) * wx * v01 + wy * (1 - wx) * v10 + wy * wx * v11
