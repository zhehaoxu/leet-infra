import torch


def solve(x: torch.Tensor) -> torch.Tensor:
    """把浮点张量量化到最近的 FP8 E4M3 可表示值再转回浮点（fake-quant 往返）。

    E4M3：1 符号 + 4 指数(bias7) + 3 尾数。规格化 step=2^(e-3)，次正规固定 step=2^-9，
    RTNE（就近、平局取偶），超过最大有限值 448 则饱和到 448。
    结果应与 x.to(torch.float8_e4m3fn).to(x.dtype) 逐元素一致。返回与 x 同形同 dtype。
    """
    # TODO: 手写 E4M3 格点量化
    raise NotImplementedError
