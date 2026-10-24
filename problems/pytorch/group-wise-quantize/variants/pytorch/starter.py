import torch


def solve(W: torch.Tensor, group_size: int, bits: int = 8) -> torch.Tensor:
    """沿最后一维分组做对称 fake-quant，返回反量化后的浮点张量（同形同 dtype）。

    qmax = 2^(bits-1)-1；每个连续组（最后一组可短）单独估 scale = 组内 absmax / qmax，
    round→clamp(-qmax,qmax)→dequant。全零组原样返回 0。
    """
    # TODO: 实现分组对称 fake-quant
    raise NotImplementedError
