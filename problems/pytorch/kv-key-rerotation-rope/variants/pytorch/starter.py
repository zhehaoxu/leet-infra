import torch


def solve(keys, indices, inv_freq):
    """KeyRerotation：gather 保留 key 后，按 (新位置-原位置) 再旋转，统一 RoPE 缓存空间。"""
    # TODO
    raise NotImplementedError
