"""参考答案：Concat —— 沿现有维拼接，其余维必须一致；用 empty + 分片写入手写法。"""

import torch


def solve(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    ts = list(tensors)
    if dim < 0:
        dim += ts[0].dim()
    out_shape = list(ts[0].shape)
    out_shape[dim] = sum(t.shape[dim] for t in ts)
    out = torch.empty(out_shape, dtype=ts[0].dtype, device=ts[0].device)
    off = 0
    for t in ts:
        sl = [slice(None)] * len(out_shape)
        sl[dim] = slice(off, off + t.shape[dim])
        out[tuple(sl)] = t
        off += t.shape[dim]
    return out
