import torch


def solve(x: torch.Tensor, axis: int) -> torch.Tensor:
    """沿 axis 做对称 per-channel fake-quant，返回反量化后的浮点张量（同形同 dtype）。

    每个通道（axis 上的一个索引）单独估计 scale = absmax/127（absmax 在其余所有维上归约），
    round→clamp(-127,127)→dequant。全零通道原样返回 0（避免除零，需与 oracle 一致）。
    需与 torch.fake_quantize_per_channel_affine(x, scale, 0, axis, -127, 127) 对齐。
    """
    # TODO: 实现 per-channel 对称 fake-quant
    raise NotImplementedError
