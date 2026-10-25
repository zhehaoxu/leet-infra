"""参考答案：全局幅值剪枝（最小的 k 个 |w| 置零，stable argsort 保证确定性）。"""

import torch


def solve(weight, sparsity):
    n = weight.numel()
    k = int(round(sparsity * n))
    idx = torch.argsort(weight.abs().flatten(), stable=True)[:k]
    flat = weight.flatten().clone()
    flat[idx] = 0.0
    return flat.view_as(weight)
