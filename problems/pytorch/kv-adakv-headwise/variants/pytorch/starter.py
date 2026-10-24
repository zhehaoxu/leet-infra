import torch


def solve(scores, compression_ratio, alpha_safeguard):
    """AdaKV：per-head 保底 → 跨头 flatten 全局 bottom-k → 返回 kept 掩码。"""
    # TODO
    raise NotImplementedError
