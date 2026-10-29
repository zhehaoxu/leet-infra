"""参考答案：k-means Lloyd 迭代 + 空簇重播种（确定性，无随机数）。

每轮：向量化平方欧氏距离 -> argmin 分配 -> 逐簇均值 -> 空簇按「到旧质心
的最近距离」降序挑未用过的点重播种 -> 最大质心位移 < tol 时停。距离与
均值在 float64 下累加，最终质心 cast 回输入 dtype，再用最终质心重算标签。
"""

import torch


def solve(points, init_centroids, tol=1e-4, max_iters=100):
    if not isinstance(points, torch.Tensor) or not isinstance(init_centroids, torch.Tensor):
        raise ValueError("points 与 init_centroids 必须是 torch.Tensor")
    if points.dim() != 2 or init_centroids.dim() != 2:
        raise ValueError("points 与 init_centroids 必须是二维张量")
    if points.dtype not in (torch.float32, torch.float64):
        raise ValueError("points 的 dtype 必须是 float32 或 float64")
    if init_centroids.dtype != points.dtype:
        raise ValueError("init_centroids 的 dtype 必须与 points 相同")
    n, d = points.shape
    k, dc = init_centroids.shape
    if n == 0:
        raise ValueError("points 至少需要一行")
    if k == 0:
        raise ValueError("init_centroids 至少需要一个质心")
    if k > n:
        raise ValueError("K 必须不大于 N")
    if d != dc:
        raise ValueError("points 与 init_centroids 的特征维必须一致")
    if isinstance(max_iters, bool) or not isinstance(max_iters, int) or max_iters < 1:
        raise ValueError("max_iters 必须是正整数")
    if tol < 0:
        raise ValueError("tol 必须非负")

    x = points.to(torch.float64)
    cent = init_centroids.to(torch.float64)
    iters = 0

    for _ in range(max_iters):
        iters += 1
        dist = ((x[:, None, :] - cent[None, :, :]) ** 2).sum(dim=2)
        labels = dist.argmin(dim=1)
        counts = torch.bincount(labels, minlength=k)

        new_cent = cent.clone()
        for j in range(k):
            if counts[j].item() > 0:
                new_cent[j] = x[labels == j].mean(dim=0)

        empty = (counts == 0).nonzero().flatten().tolist()
        if empty:
            nearest_old = dist.min(dim=1).values.tolist()
            order = sorted(range(n), key=lambda i: (-nearest_old[i], i))
            used = set()
            for j in empty:
                for i in order:
                    if i not in used:
                        used.add(i)
                        new_cent[j] = x[i]
                        break

        disp = torch.sqrt(((new_cent - cent) ** 2).sum(dim=1)).max().item()
        cent = new_cent
        if disp < tol:
            break

    final_cent = cent.to(points.dtype)
    final_dist = (
        (points.to(torch.float64)[:, None, :] - final_cent.to(torch.float64)[None, :, :]) ** 2
    ).sum(dim=2)
    final_labels = final_dist.argmin(dim=1).to(torch.int64)
    return final_labels, final_cent, iters
