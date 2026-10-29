import torch


def solve(W):
    """BlockFP：1×128 tile + UE8M0 共享指数编码。

    返回 (scale_codes, normalized)。
    """
    # TODO: 按 1×128 tile 求 absmax、编码 UE8M0、并返回归一化值
    raise NotImplementedError
