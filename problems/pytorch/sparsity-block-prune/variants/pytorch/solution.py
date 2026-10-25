"""参考答案：块结构剪枝（整块按 L1 范数全局排序、剪最小的若干整块）。"""

import torch


def solve(weight, sparsity, block):
    r, c = weight.shape
    nb = c // block
    blocks = weight.reshape(r, nb, block)
    norms = blocks.abs().sum(dim=2).flatten()
    total = r * nb
    k = int(round(sparsity * total))
    idx = norms.argsort(stable=True)[:k]
    mask = torch.ones(total, dtype=torch.float32)
    mask[idx] = 0.0
    return (blocks * mask.view(r, nb, 1)).reshape(r, c)
