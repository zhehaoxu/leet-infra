"""参考答案：searchsorted 分桶 + bincount 直方图（纯比较运算实现，不调原生）。

口径：right=False 时 idx = #{b < x}（最小 i 使 b[i] >= x）；right=True 时
idx = #{b <= x}（最小 i 使 b[i] > x）。与 torch.searchsorted 的
side="left"/"right" 逐元素一致（实测确认，含重复 boundary 与空 boundaries）。
counts 用 index_add_ 按 idx 累加，长度恒为 M+1。
"""

import torch


def solve(boundaries, x, right=False, weights=None):
    if boundaries.dim() != 1:
        raise RuntimeError(
            "boundaries tensor should be 1-dimensional, got "
            f"{boundaries.dim()} dimensions")
    m = boundaries.numel()
    xf = x.reshape(-1)
    bc = boundaries.unsqueeze(0)      # (1, M)
    xc = xf.unsqueeze(1)              # (N, 1)
    # left：严格小于 x 的 boundary 个数；right：<= x 的个数
    if right:
        idx = (bc <= xc).sum(dim=1)
    else:
        idx = (bc < xc).sum(dim=1)
    idx = idx.to(torch.long).reshape(x.shape)

    if weights is not None:
        if weights.shape != x.shape:
            raise ValueError(
                f"weights 必须与 x 同形状，got {tuple(weights.shape)} vs {tuple(x.shape)}")
        wf = weights.reshape(-1)
        if not wf.is_floating_point():
            wf = wf.to(torch.float32)
        counts = torch.zeros(m + 1, dtype=wf.dtype)
        counts.index_add_(0, idx.reshape(-1), wf)
    else:
        counts = torch.zeros(m + 1, dtype=torch.int64)
        counts.index_add_(0, idx.reshape(-1),
                          torch.ones(idx.numel(), dtype=torch.int64))
    return idx, counts
