import torch


def solve(W):
    """MX Microscaling：32 元素块 + E8M0 共享指数编码。

    返回 (scale_codes, normalized)。
    """
    # TODO: 按 32 元素块求 absmax、编码 E8M0、并返回归一化值
    raise NotImplementedError
