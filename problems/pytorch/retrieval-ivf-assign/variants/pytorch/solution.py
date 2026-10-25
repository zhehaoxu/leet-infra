"""参考答案：广播计算平方欧氏距离矩阵后逐行 argmin。"""

import torch


def solve(x, centroids):
    # (n,1,D) - (1,k,D) -> (n,k,D) -> 逐维平方和 -> (n,k)
    d2 = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(dim=-1)
    return d2.argmin(dim=1).to(torch.long)
