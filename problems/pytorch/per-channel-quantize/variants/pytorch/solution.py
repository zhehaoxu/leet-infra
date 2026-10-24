import torch


def solve(x: torch.Tensor, axis: int) -> torch.Tensor:
    axis = axis % x.dim()
    reduce_dims = tuple(i for i in range(x.dim()) if i != axis)
    # absmax 在除 axis 外的所有维上归约，得到每通道一个 scale
    absmax = x.abs().amax(dim=reduce_dims) if reduce_dims else x.abs()
    scale = absmax / 127.0
    # 广播形状：把 axis 维保留、其余置 1
    bshape = [1] * x.dim()
    bshape[axis] = x.size(axis)
    scale = scale.view(bshape)
    # 全零通道 scale=0 → 该通道输出 0（避免除零，与 oracle 对齐）
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(x / safe_scale), -127, 127)
    out = q * scale
    return out
