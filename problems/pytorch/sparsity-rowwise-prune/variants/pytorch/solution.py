"""参考答案：逐行局部幅值剪枝（每行各自剪最小 k 个 |w|）。"""

import torch


def solve(weight, sparsity):
    r, c = weight.shape
    k = int(round(sparsity * c))
    idx = weight.abs().argsort(dim=1, stable=True)[:, :k]
    out = weight.clone()
    out.scatter_(1, idx, 0.0)
    return out
