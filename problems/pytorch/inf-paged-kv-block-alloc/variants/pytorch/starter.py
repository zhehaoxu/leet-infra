import torch


def solve(free_blocks, block_size, seq_lens):
    """PagedAttention 准入分配：每序列取 ceil(len/block_size) 个空闲块（从头部），
    空闲不足以整条满足则记 []。返回 (block_tables, remaining_free)，均为 list。"""
    # TODO
    raise NotImplementedError
