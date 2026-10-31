"""参考答案：先算 num_samples，再拼出 effective，最后跨步切片。

四个口径点：

1. **步数公式的分支条件不是 `drop_last`，而是「`drop_last` 且除不尽」**。整除时两条公式重合，
   写成分支式是为了对齐上游那句
   `math.ceil((len(dataset) - num_replicas) / num_replicas)`——它在 N = qR + r（0 < r < R）时恰等于
   floor(N/R)，在 N < R 时恰等于 0，所以本实现直接用整除表达，不引入浮点。
2. **补齐是从头抄，不是随机补，也不是补到最后一个 rank**。抄法有两种（`padding <= N` 时接
   `order[:padding]`，否则把整个 `order` 重复若干遍再截断），但**结果等价于** `effective[k] = order[k % N]`，
   所以这里写成显式拼接、oracle 写成取模，两条路径必须撞在一起。
3. **切片是跨步的，不是连续块**。`effective[rank:total:R]` 意味着 rank 拿到的是第
   rank, rank+R, rank+2R… 个槽位。答成「每人一段连续区间」是这道题最常见的错法——它只在
   `num_replicas == 1` 时成立，且会把「哪几个样本被重复」整个搞错。
4. **coverage 统计的是 effective，不是 shard**。它是全局账（所有 rank 加起来），不是当前 rank 的账。

`num_samples == 0` 是完全合法的分支（$N<R$ 且砍尾），此时 `shard` 是空的 int64 张量而不是 `None`。
"""

import torch


def _check(order, num_replicas, rank, drop_last):
    if not isinstance(order, torch.Tensor):
        raise ValueError("order must be a torch.Tensor")
    if order.device.type != "cpu":
        raise ValueError("order must be on CPU")
    if order.dtype != torch.int64:
        raise ValueError("order must be int64")
    if order.dim() != 1 or order.numel() < 1:
        raise ValueError("order must be a non-empty 1-D tensor")
    if not isinstance(drop_last, bool):
        raise ValueError("drop_last must be a python bool")
    for name, value in (("num_replicas", num_replicas), ("rank", rank)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be a python int")
    if num_replicas < 1:
        raise ValueError("num_replicas must be at least 1")
    if rank < 0 or rank >= num_replicas:
        raise ValueError("rank must be in [0, num_replicas)")
    size = order.numel()
    values = order.tolist()
    if min(values) < 0 or max(values) >= size:
        raise ValueError("order must hold indices within [0, N)")
    if len(set(values)) != size:
        raise ValueError("order must be a permutation of 0..N-1")


def solve(order, num_replicas, rank, drop_last):
    _check(order, num_replicas, rank, drop_last)
    size = order.numel()
    indices = order.tolist()

    if drop_last and size % num_replicas != 0:
        num_samples = size // num_replicas
    else:
        num_samples = -(-size // num_replicas)
    total_size = num_samples * num_replicas

    if drop_last:
        effective = indices[:total_size]
    else:
        padding = total_size - size
        if padding <= size:
            effective = indices + indices[:padding]
        else:
            repeats = -(-padding // size)
            effective = indices + (indices * repeats)[:padding]

    shard = effective[rank:total_size:num_replicas]
    coverage = [0] * size
    for value in effective:
        coverage[value] += 1

    return (torch.tensor(shard, dtype=torch.int64), num_samples, total_size,
            torch.tensor(coverage, dtype=torch.int64))
