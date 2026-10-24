import torch


def solve(attn, num_kv_heads):
    """TOVA：最后 query 的注意力跨头平均 → 复制 num_kv_heads 份 + 尾部 pad max+1。"""
    # TODO
    raise NotImplementedError
