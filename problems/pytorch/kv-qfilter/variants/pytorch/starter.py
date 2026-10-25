import torch


def solve(q_filters, keys, compression_ratio):
    """Q-Filter：scores[n,h,t] = -Σ_d q_filters[n,h,d]·keys[n,h,t,d]（负点积，无√D缩放）；
    kept=每 (n,h) 行 top max(1,int(L*(1-cr)))（平局小位置优先）升序。
    返回 (scores float32 (N,H,L), kept int64 (N,H,n_kept))。q_filters (N,H,D)，keys (N,H,L,D)，CPU float32。"""
    # TODO
    raise NotImplementedError
