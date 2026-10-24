import torch


def solve(attn, kernel_size):
    """SnapKV：窗口 attn 沿 W 求均 → avg_pool1d → 尾部 W 位以 max+1 拼接。返回 (H, P+W)。"""
    # TODO
    raise NotImplementedError
