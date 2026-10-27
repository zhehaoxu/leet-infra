"""参考答案：scatter_reduce 聚合写入（稳定排序分组 + 按贡献顺序串行聚合）。

口径要点（全部经原生 CPU 行为实测确认）：
- mean 是 Welford 型增量更新，不是 batch 的「总和/总数」——重复元素多时
  两者在浮点下差异显著（如 base=0、src=[1e8,1,1,1] 时原生 mean=2e7，
  batch 公式给 2.5e7）；
- include_self=False 时被触及的位置从首个 scatter 贡献起算；
- 未被触及的位置恒等于 input 原值（两种 include_self 都如此）；
- 同一位置的贡献按其 dim 维坐标升序到达（行主序遍历下的必然结果），
  因此用稳定排序分组后组内天然保持该顺序，串行聚合可与原生舍入一致。
"""

import torch

_REDUCES = ("sum", "prod", "mean", "amax", "amin")


def _fold(acc, vals, reduce):
    """把该位置收到的贡献 vals（已按到达顺序排好）串行并入标量张量 acc。

    acc 已含初值（include_self=True 的 input 原值，或 include_self=False 的
    首个贡献）；mean 的计数从 acc 已代表 1 个值开始递增。逐个用标量张量运算
    更新，保证与原生一致的 dtype 舍入行为。
    """
    n = 1
    for v in vals:
        n += 1
        if reduce == "sum":
            acc = acc + v
        elif reduce == "prod":
            acc = acc * v
        elif reduce == "amax":
            acc = torch.maximum(acc, v)
        elif reduce == "amin":
            acc = torch.minimum(acc, v)
        else:  # mean：Welford 型增量更新
            acc = acc + (v - acc) / n
    return acc


def solve(input, src, index, dim=0, reduce="sum", include_self=True):
    if reduce not in _REDUCES:
        raise ValueError(
            f"unknown reduce: {reduce!r}（合法取值 sum/prod/mean/amax/amin）")
    if not (input.dim() == src.dim() == index.dim()):
        raise RuntimeError(
            "Index tensor must have the same number of dimensions as self tensor")
    if index.is_floating_point():
        raise RuntimeError("scatter(): Expected index to be an integer tensor")
    if src.dtype != input.dtype:
        raise RuntimeError("scatter(): Expected self.dtype to be equal to src.dtype")

    ndim = input.dim()
    d = dim + ndim if dim < 0 else dim
    if not 0 <= d < ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim - 1}], but got {dim})")
    for dd in range(ndim):
        if index.size(dd) > src.size(dd):
            raise RuntimeError(
                f"Expected index size at dim {dd} to be no larger than src size "
                f"{src.size(dd)}, but got {index.size(dd)}")
        if dd != d and index.size(dd) > input.size(dd):
            raise RuntimeError(
                f"Expected index size at dim {dd} to be no larger than self size "
                f"{input.size(dd)} apart from dimension {d}")

    # 越界检查（含负值），与原生同样抛 RuntimeError
    if index.numel():
        idx_flat0 = index.reshape(-1)
        bad = (idx_flat0 < 0) | (idx_flat0 >= input.size(d))
        if bool(bad.any()):
            v = int(idx_flat0[bad][0])
            raise RuntimeError(
                f"index {v} is out of bounds for dimension {d} with size {input.size(d)}")

    # index 的行主序扁平序号 → input 的扁平位置（dim 维坐标换成 index 值）
    n_idx = index.numel()
    out = input.reshape(-1).clone()  # 未被触及的位置保持 input 原值
    if n_idx == 0:
        return out.reshape(input.shape)

    sizes = index.shape
    strides_in = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides_in[i] = strides_in[i + 1] * input.shape[i + 1]
    lin = torch.arange(n_idx)
    pos = torch.zeros(n_idx, dtype=torch.long)
    idx_flat = index.reshape(-1).to(torch.long)
    after = 1
    for dd in range(ndim - 1, -1, -1):
        coord = (lin // after) % sizes[dd]
        if dd == d:
            pos += idx_flat * strides_in[dd]
        else:
            pos += coord * strides_in[dd]
        after *= sizes[dd]

    # src 与 index 同坐标配对；src 更大时取 index 形状的前缀
    if src.shape == index.shape:
        vals = src.reshape(-1)
    else:
        vals = src[tuple(slice(0, s) for s in sizes)].reshape(-1)

    # 稳定排序分组：组内保持行主序（即 dim 维坐标升序的贡献顺序）
    order = torch.argsort(pos, stable=True)
    pos_s, val_s = pos[order], vals[order]
    starts = torch.ones(n_idx, dtype=torch.bool)
    starts[1:] = pos_s[1:] != pos_s[:-1]
    bounds = torch.nonzero(starts).flatten().tolist() + [n_idx]
    for a, b in zip(bounds[:-1], bounds[1:]):
        p = int(pos_s[a])
        seg = val_s[a:b]
        if include_self:
            acc, rest = out[p], seg
        else:
            acc, rest = seg[0], seg[1:]
        out[p] = _fold(acc, rest, reduce)
    return out.reshape(input.shape)
