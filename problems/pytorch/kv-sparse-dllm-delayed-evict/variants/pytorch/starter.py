import torch


def solve(k, q_block, bef_filtered_len, block_len, keep_ratio=0.5, pool_kernel_size=3):
    """Sparse-dLLM：剔除当前 block 后 block 内 query 均值打分 → 跨 head 均值 → max_pool1d → topk。"""
    # TODO
    raise NotImplementedError
