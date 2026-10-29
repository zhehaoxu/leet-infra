"""参考答案：头级结构化剪枝（Frobenius 权重范数 × 校准激活范数，全局取最低的若干头）。"""

import torch


def solve(weight, calib, sparsity):
    l, h = weight.shape[0], weight.shape[1]
    w_norm = weight.flatten(2).norm(dim=2)
    x_norm = calib.flatten(2).norm(dim=2)
    importance = w_norm * x_norm
    total = l * h
    k = int(round(sparsity * total))
    drop = torch.argsort(importance.flatten(), stable=True)[:k]
    keep = torch.ones(total, dtype=torch.float32)
    keep[drop] = 0.0
    pruned = weight * keep.view(l, h, 1, 1)
    return pruned, importance
