import torch


def solve(W, block_size):
    """DuQuant 的 ZigZag 通道重排。

    按列的最大绝对权重从大到小排序，再把它们「蛇形」轮流分配进
    H // block_size 个块，最后对每个块内部再按同一键降序排序，
    按块拼接得到通道置换。
    """
    if not isinstance(W, torch.Tensor) or W.dim() != 2 or not W.is_floating_point():
        raise ValueError("W 必须是二维浮点张量")
    if W.shape[0] == 0 or W.shape[1] == 0:
        raise ValueError("W 的行数和列数都必须大于 0")
    if not bool(torch.isfinite(W).all()):
        raise ValueError("W 必须有限")
    if not isinstance(block_size, int) or isinstance(block_size, bool):
        raise ValueError("block_size 必须是整数")
    if block_size < 1 or block_size > W.shape[1]:
        raise ValueError("block_size 必须在 1 到列数之间")
    if W.shape[1] % block_size != 0:
        raise ValueError("列数必须能被 block_size 整除")

    hidden = W.shape[1]
    groups = hidden // block_size

    weight_max = W.to(torch.float64).abs().amax(dim=0)
    pairs = [(c, float(weight_max[c])) for c in range(hidden)]
    pairs.sort(key=lambda item: item[1], reverse=True)

    buckets = [[] for _ in range(groups)]
    cur = 0
    up = True
    for pair in pairs:
        buckets[cur].append(pair)
        if up:
            cur += 1
            if cur == groups:
                cur -= 1
                up = False
        else:
            cur -= 1
            if cur == -1:
                cur += 1
                up = True

    perm = []
    for bucket in buckets:
        bucket.sort(key=lambda item: item[1], reverse=True)
        perm.extend(idx for idx, _ in bucket)

    perm_t = torch.tensor(perm, dtype=torch.int64)
    return {
        "perm": perm_t,
        "permuted": W[:, perm_t].to(torch.float32),
    }
