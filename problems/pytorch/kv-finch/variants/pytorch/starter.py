import torch


def solve(attn, num_kv_heads, compression_ratio):
    """FINCH：attn (Hq,W,C) 窗口query→上下文key注意力(行0=最老query)；每行 i 乘 (C+i)，
    窗口维平均→GQA组平均→(H,C)，右 pad W 个「全局max+1」得 scores (H,L=C+W)；
    kept=每行 top max(1,int(L*(1-cr)))（平局小位置优先）升序。返回 (scores float32 (H,L), kept int64 (H,n_kept))。"""
    # TODO
    raise NotImplementedError
