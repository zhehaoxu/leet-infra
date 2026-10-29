"""参考答案：2:4 压缩值 + 4-bit 偏移元数据（每 8 组对齐成一个 32-bit 字）。"""

import torch


def solve(weight):
    r, k = weight.shape
    groups = k // 4
    g = weight.reshape(r, groups, 4)
    order = torch.argsort((g == 0).to(torch.int8), dim=2, stable=True)
    idx = order[:, :, :2]
    values = torch.gather(g, 2, idx).reshape(r, k // 2)

    i0 = idx[:, :, 0].to(torch.int64)
    i1 = idx[:, :, 1].to(torch.int64)
    nib = i0 + ((i1 - 1) << 2)

    padded = ((groups + 7) // 8) * 8
    table = torch.zeros(r, padded, dtype=torch.int64)
    table[:, :groups] = nib
    meta = (table[:, 0::2] | (table[:, 1::2] << 4)).to(torch.uint8)
    return values, meta
